# Pipeline v3.1 性能优化分析报告

## 背景

基于v2.x版本的实际运行数据（处理一本200页技术教材约需**7小时**），对以下两个核心瓶颈进行分析并提出优化方案。

---

## 问题一：执行效率与Token消耗

### 瓶颈定位

v2.x版本对每个chunk的处理分为两个**独立的API调用阶段**：

```
每个 chunk 的处理流程（v2.x）：
  ├── [规则引擎过滤]   → 无API调用，快速
  ├── API Call 1: diagnose_chunk()   → 返回诊断类型 + sub_types
  └── API Call 2×N: generate_samples() → 每种 sub_type 独立一次调用
```

**时间估算（v2.x，300个chunk示例）：**

| 环节 | 耗时估算 |
|---|---|
| 每次API调用延迟 | 3~15秒 |
| 每chunk平均API调用次数 | 2.5次 |
| batch_delay（1秒/chunk） | 300秒 |
| **300个chunk总计** | **约4~7小时** |

### 优化方案：合并诊断+生成为单次调用

将"内容诊断"和"定向生成"合并进一个精心设计的Prompt，让LLM在**一次调用**中完成：

```
每个 chunk 的处理流程（v3.1）：
  ├── [规则引擎过滤]   → 无API调用，快速
  └── API Call 1: diagnose_and_generate_merged()
                       → 同时返回：诊断类型 + 选定语料类型 + 生成的样本
```

**效果对比：**

| 指标 | v2.x | v3.1 | 改善幅度 |
|---|---|---|---|
| 每chunk API调用次数 | 2~3次 | **1次** | **减少约50%** |
| batch_delay | 1.0秒 | 0.5秒 | 减少50% |
| 300 chunk预计耗时（单线程） | 4~7小时 | **2~3.5小时** | 约减半 |

### 合并Prompt设计要点

合并后的`SYSTEM_PROMPT_MERGED`在一个JSON输出中同时包含：
- `diagnosis`：内容诊断类型
- `confidence`：置信度
- `selected_types`：自动选择的语料类型列表
- `samples`：生成的所有样本（支持多种类型混合输出）

这样LLM只需一次"思考-输出"循环，避免了两次独立调用的冷启动开销和通信往返延迟。

### 关于metadata冗余的说明

v2.x版本每个SFTSample中保存了大量诊断元数据（filter_stage、diagnosis_reason等），这些数据：
- **不是导致7小时的主因**（IO开销可忽略）
- **主因是API调用次数**

v3.1在metadata中只保留必要字段（chapter、section、context_anchor、diagnosis_reason、knowledge_density、matched_categories），去掉了filter_stage等调试专用字段，使每条样本更简洁。

---

## 问题二：多线程并发执行

### 问题分析

**结论：合理且高度必要，与优化一互补，叠加后效果最显著。**

v2.x的`process_chunks()`是严格单线程的`for`循环，无法利用API服务端的并发处理能力。DashScope（Qwen3）的API服务是完全并发的，单账号通常支持5~20个并发请求（取决于套餐），但v2.x始终只用了1个并发。

### 优化方案：ThreadPoolExecutor并行处理

v3.1使用`concurrent.futures.ThreadPoolExecutor`实现多线程并行：

```python
# v3.1并行处理示意
with ThreadPoolExecutor(max_workers=N) as executor:
    futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
    for future in as_completed(futures):
        # 收集结果，线程安全地合并统计
```

**效果叠加估算（合并调用 + 4线程并发）：**

| 配置 | 300 chunk预计耗时 |
|---|---|
| v2.x（单线程，2次调用） | 4~7 小时 |
| v3.1（单线程，1次调用） | 2~3.5 小时 |
| v3.1（4线程，1次调用） | **0.5~1 小时** |

**加速比理论上限：** min(API并发上限, max_workers) × 单次调用优化系数

### 线程安全设计

v3.1中所有`self.stats`的读写操作均通过`threading.Lock`保护：

```python
def _incr_stat(self, key: str, amount: int = 1):
    with self._stats_lock:          # 获取锁
        self.stats[key] += amount   # 原子更新
```

关键线程安全点：
- `total_api_calls` / `total_tokens_used`：每次API调用后更新
- `chunks_filtered` / `chunks_passed`：每次过滤决策后更新
- `type_counts`：每条样本生成后更新

### 并发参数建议

| API服务 | 建议 max_workers | 说明 |
|---|---|---|
| DashScope Qwen3（免费额度） | 2~3 | QPM限制较低，保守起见 |
| DashScope Qwen3（付费套餐） | 4~8 | 根据账号QPM上限调整 |
| OpenAI GPT-4系列 | 3~5 | 注意TPM限制 |
| 本地部署（vLLM等） | 8~16 | 受GPU显存和并发能力限制 |

### 使用方法

```bash
# 单线程（默认，兼容旧版行为）
python main.py --input data/book.md --mode sft --api-key sk-xxx

# 4线程并行
python main.py --input data/book.md --mode sft --api-key sk-xxx --workers 4

# 冷启动测试（30个chunk，4线程）
python main.py --input data/book.md --mode full --api-key sk-xxx --cold-start 30 --workers 4
```

---

## CPT生成器评估

经代码审查，`cpt_generator.py` **无需修改**，原因如下：

1. **类型系统独立**：CPT生成器不使用任何SFT语料类型名称，仅操作`Chunk`对象的原始内容字段
2. **无API调用**：CPT生成是纯文本处理，没有LLM调用，不存在效率问题
3. **设计合理**：基础模式（直接转换chunk）和上下文模式（滑窗合并）均符合CPT训练数据的设计原则
4. **对所有chunk生成语料是正确行为**：CPT的目标是让模型"阅读"全书内容，包括前言、目录等低价值内容（SFT才需要过滤）

---

## 总结

| 优化项 | 实现位置 | 预期收益 |
|---|---|---|
| 合并诊断+生成（单次API调用） | `src/sft_generator.py` | API调用减少~50%，Token消耗减少~30% |
| 多线程并发（ThreadPoolExecutor） | `src/sft_generator.py` | 按workers数线性加速（受API限流约束） |
| 线程安全统计（threading.Lock） | `src/sft_generator.py` | 确保多线程数据一致性 |
| batch_delay从1.0s降至0.5s | `config.py` `LLMConfig` | 顺序模式总等待时间减半 |
| 精简样本metadata | `src/sft_generator.py` | 输出文件更简洁，减少存储 |

**两项优化叠加预期效果：7小时 → 约0.5~1小时**（以4线程、300个chunk为例）

---

*本文档由 Claude Sonnet 4.6 基于代码审查自动生成 | Pipeline v3.1*
