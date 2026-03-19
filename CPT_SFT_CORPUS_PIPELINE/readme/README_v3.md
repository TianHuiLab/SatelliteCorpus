# 高质量动态微调语料生成Pipeline v3.1

航天领域高质量SFT语料自动化生产工具。从结构化Markdown书籍文件中，经过智能分块、知识密度过滤和LLM定向生成，产出可直接用于大模型训练的高质量微调语料。

**v3.1 核心升级**：在v2.x基础上，将"内容诊断"与"语料生成"合并为单次LLM调用，并新增多线程并发支持，处理效率提升约**5~10倍**。

---

## 核心特性

### v3.1 新特性
- **合并调用优化**：`diagnose_and_generate_merged()` 将原来的2次API调用合并为1次，减少约50%的API请求和Token消耗
- **多线程并发**：`--workers N` 参数启用 `ThreadPoolExecutor` 并行处理，按线程数线性加速
- **线程安全统计**：所有统计字典通过 `threading.Lock` 保护，确保并发下数据一致性

### 沿用特性
- **智能语义分块**：遵循"最小语义单元"原则，200-800字分块，正确处理HTML表格、LaTeX公式
- **参考文献分离**：自动检测并分离章节末尾参考文献区域
- **知识密度过滤**：规则引擎预过滤 + LLM精细过滤双重机制
- **6种语料类型**：`fact_qa`、`comprehension`、`reasoning_qa`、`summarization`、`information_extraction`、`calculation`
- **自包含约束**：严格过滤含书籍结构指代、模糊指代词、图表编号等违规样本
- **自动化报告**：每次运行生成详细Markdown报告

---

## 语料类型体系（v3.x）

| 类型标识 | 中文名 | 适合内容 | 示例问题形式 |
|---|---|---|---|
| `fact_qa` | 基础事实问答 | 定义型内容 | "是什么"、"有哪些"、"定义是什么" |
| `comprehension` | 理解型问题 | 过程型/原理型内容 | "为什么"、"作用是什么"、"如何工作" |
| `reasoning_qa` | 推理问题 | 案例型/因果型内容 | 需要"步骤1/步骤2/结论"推理链 |
| `summarization` | 总结任务 | 描述型内容 | "请总结下面这段描述：..." |
| `information_extraction` | 信息抽取 | 数据型/参数型内容 | "关键技术参数有哪些"、"系统组成是什么" |
| `calculation` | 计算问题 | 含公式的内容 | 含完整LaTeX公式推导过程 |

**内容诊断 → 语料类型对应关系：**
```
定义型 → fact_qa
案例型 → reasoning_qa
过程型 → comprehension
描述型 → summarization
数据型 → information_extraction
含公式 → calculation（追加）
```

---

## 项目结构

```
CPT_SFT_CORPUS_PIPELINE/
├── main.py                   # 主控流程（v3.1）
├── config.py                 # 全局配置（含 max_workers）
├── src/
│   ├── __init__.py
│   ├── chunker.py            # 智能语义分块器 v2.1
│   ├── cpt_generator.py      # CPT语料生成器（无需API，稳定）
│   ├── data_loader.py        # Markdown加载与预处理
│   ├── knowledge_filter.py   # 知识密度过滤器 v2.1（双重过滤）
│   ├── sft_generator.py      # SFT语料生成器 v3.1（合并调用+多线程）
│   └── utils.py              # 工具函数
├── readme/
│   ├── README_v3.md          # 本文档
│   └── optimization_analysis.md  # 性能优化分析报告
└── .vscode/
    └── launch.json           # VS Code 调试配置
```

---

## 快速开始

### 1. 环境依赖

```bash
pip install openai tqdm
```

### 2. 基础用法

```bash
# 仅生成CPT语料（无需API）
python main.py --input data/book.md --output output/ --mode cpt

# 仅生成SFT语料（需要API Key）
python main.py \
  --input data/book.md \
  --output output/ \
  --mode sft \
  --api-key sk-xxx

# 同时生成CPT+SFT（完整模式）
python main.py \
  --input data/book.md \
  --output output/ \
  --mode full \
  --api-key sk-xxx
```

### 3. 性能优化用法（v3.1新增）

```bash
# 4线程并行（推荐，速度提升约4倍）
python main.py \
  --input data/book.md \
  --mode sft \
  --api-key sk-xxx \
  --workers 4

# 冷启动测试（前30个chunk，快速验证）
python main.py \
  --input data/book.md \
  --mode full \
  --api-key sk-xxx \
  --cold-start 30 \
  --workers 3

# 禁用LLM过滤（节省API调用，适合快速批量处理）
python main.py \
  --input data/book.md \
  --mode sft \
  --api-key sk-xxx \
  --no-llm-filter \
  --workers 4
```

---

## 命令行参数完整说明

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` / `-i` | 必填 | 输入MD文件路径或目录路径 |
| `--output` / `-o` | `./output` | 输出目录路径 |
| `--mode` / `-m` | `full` | 运行模式：`cpt` / `sft` / `full` |
| `--config` / `-c` | — | JSON配置文件路径（可覆盖所有参数） |

### API参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--api-key` | — | LLM API Key（优先级高于环境变量） |
| `--model` | `qwen3-max` | 模型名称 |
| `--base-url` | DashScope URL | API Base URL |
| `--workers` | `1` | **v3.1新增**：并发线程数（建议3~5） |

### 数据处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cold-start` | `0` | 冷启动：仅处理前N个chunk（0=全部） |
| `--min-chars` | `200` | 最小分块字符数 |
| `--max-chars` | `800` | 最大分块字符数 |
| `--domain` | `aerospace` | 领域标识（注入Prompt） |
| `--format` | `sharegpt` | SFT输出格式：`sharegpt` / `alpaca` / `messages` |
| `--glossary` | — | 领域术语表JSON文件路径 |

### 过滤参数

| 参数 | 说明 |
|---|---|
| `--no-filter` | 禁用所有知识密度过滤（所有chunk均生成） |
| `--no-llm-filter` | 仅用规则引擎过滤，不调用LLM（节省API，速度更快） |

---

## 输出文件结构

每次运行在 `output/<时间戳>/` 下生成：

```
output/20250312_143000/
├── pipeline.log              # 运行日志
├── config_snapshot.json      # 本次配置快照
├── chunks.jsonl              # 所有分块结果
├── cpt_<书名>.jsonl          # CPT语料（基础模式）
├── cpt_contextual_<书名>.jsonl # CPT语料（上下文模式）
├── filter_details.jsonl      # 每个chunk的过滤决策详情
├── filtered_chunks.jsonl     # 被过滤的低价值chunk
├── diagnoses.jsonl           # 每个通过过滤chunk的诊断结果
├── sft_train.sharegpt.jsonl  # SFT语料（ShareGPT格式，主要训练集）
├── sft_train.alpaca.jsonl    # SFT语料（Alpaca格式，同步输出）
├── sft_review_pool.jsonl     # 低置信度样本（人工审核池）
├── pipeline_report.json      # JSON格式运行报告
└── pipeline_report.md        # Markdown格式可读报告
```

---

## 技术架构详解

### Pipeline执行流程（v3.1）

```
MD文件输入
    │
    ▼
[Step 1] 文档加载（MarkdownLoader）
    │  → 解析章节结构、标题层级、参考文献标记
    │
    ▼
[Step 2] 语义分块（SemanticChunker v2.1）
    │  → 200~800字分块，保留上下文锚点
    │  → 处理HTML表格、LaTeX公式块
    │
    ▼
[Step 3] CPT语料生成（CPTGenerator）
    │  → 无API调用，直接输出
    │  → 基础模式 + 上下文模式
    │
    ▼
[Step 4] 知识密度预过滤（KnowledgeDensityFilter v2.1）
    │  → 规则引擎：快速拦截出版信息、参考文献等低价值内容
    │  → LLM精细判断（仅对规则引擎无法确定的内容，可选）
    │
    ▼
[Step 5] SFT语料生成（SFTGenerator v3.1）✨ 优化核心
    │  → 【v3.1】单次API调用完成：内容诊断 + 类型选择 + 样本生成
    │  → 【v3.1】ThreadPoolExecutor并行（workers=N）
    │  → 自包含约束检查（过滤书籍结构指代等）
    │
    ▼
[Step 6~7] 报告生成与保存
```

### v3.1 合并调用设计（SYSTEM_PROMPT_MERGED）

合并Prompt要求LLM在一次输出中完成：
1. **诊断**：判断文本属于定义型/案例型/过程型/描述型/数据型
2. **选型**：从6种语料类型中选择1~2种最合适的
3. **生成**：直接输出对应类型的样本，统一在 `samples[]` 数组中

输出格式示例：
```json
{
  "diagnosis": "定义型",
  "confidence": 0.88,
  "reason": "文本包含多个术语定义和参数说明",
  "selected_types": ["fact_qa", "information_extraction"],
  "samples": [
    {"type": "fact_qa", "question": "卫星姿态确定系统的主要功能是什么？", "answer": "..."},
    {"type": "information_extraction", "question": "姿态控制系统的主要组成部分有哪些？", "answer": "..."}
  ]
}
```

---

## 性能基准

以处理《微小卫星总体设计与工程实践》（200页，约350个有效chunk）为基准：

| 配置 | 预计耗时 | API调用次数 |
|---|---|---|
| v2.x 单线程 | 5~7 小时 | ~800次 |
| v3.1 单线程（`--workers 1`） | 2~3 小时 | ~400次 |
| v3.1 三线程（`--workers 3`） | 45~70 分钟 | ~400次 |
| v3.1 五线程（`--workers 5`） | 30~50 分钟 | ~400次 |

> 注：实际耗时受API响应延迟、网络状况和账号限流影响，上述为估算值。

---

## 配置文件（config.py）关键参数

```python
LLMConfig(
    model="qwen3-max",
    batch_delay=0.5,     # 顺序模式下chunk间延迟（v3.1从1.0降至0.5）
    max_workers=1,       # 并发线程数（--workers参数覆盖此值）
)

SFTConfig(
    max_samples_per_chunk=3,    # 每个chunk生成的样本数（合并调用下建议2~3）
    confidence_threshold=0.7,   # 低于此值进入人工审核池
    target_type_distribution={  # 目标语料类型分布（用于报告分析）
        "fact_qa": 0.25,
        "comprehension": 0.15,
        "reasoning_qa": 0.25,
        "summarization": 0.15,
        "information_extraction": 0.15,
        "calculation": 0.05,
    }
)
```

---

## 版本历史

| 版本 | 主要变化 |
|---|---|
| v3.1 | 合并诊断+生成为单次API调用；多线程并发支持；线程安全统计 |
| v3.0 | 更新语料类型体系（fact_qa/comprehension/reasoning_qa等6种新类型） |
| v2.2 | 融合v1完整类型体系；参考文献智能分离；三阶段SFT生成 |
| v2.1 | 智能语义分块增强；HTML表格/LaTeX公式处理；参考文献内容级检测 |
| v2.0 | 知识密度过滤（规则引擎+LLM双重）；内容诊断路由 |
| v1.0 | 初始版本：6种原始语料类型；基础API调用 |

---

*本README由 Claude Sonnet 4.6 基于代码审查自动生成 | Pipeline v3.1*
