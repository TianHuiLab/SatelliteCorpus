# 高质量动态微调语料生成Pipeline v3.1

从结构化Markdown书籍文件中，自动生成高质量、多样化微调语料的Python Pipeline。专为技术类书籍设计，将书籍内容转化为可直接用于训练领域大模型的SFT（Supervised Fine-Tuning）与CPT（Continual Pre-Training）语料。

**v3.1 核心升级**：将原来每个chunk需要2次LLM调用的"诊断→生成"两阶段流程，合并为**单次调用**完成；同时新增**多线程并发**支持，实测处理效率提升约5~10倍（7小时 → 约40分钟）。

---

## 版本更新说明

| 版本 | 主要变化 |
|---|---|
| **v3.1（当前）** | 合并诊断+生成为单次API调用；多线程并发（`--workers`）；线程安全统计；list/str类型防御修复 |
| v3.0 | 更新语料类型体系为6种新类型（fact_qa / comprehension / reasoning_qa等） |
| v2.2 | 参考文献智能分离；三阶段SFT生成；融合v1完整类型体系 |
| v2.1 | 语义分块增强；HTML表格/LaTeX公式处理；参考文献内容级检测 |
| v2.0 | 知识密度过滤（规则引擎+LLM双重）；内容诊断路由 |

---

## 核心特性

### v3.1 新增
- **合并调用**（`SYSTEM_PROMPT_MERGED`）：内容诊断与语料生成在一个Prompt内完成，每个chunk只需1次API调用，较v2.x减少约50%的请求次数和Token消耗
- **多线程并发**：`--workers N` 启用 `ThreadPoolExecutor`，多个chunk同时向API发送请求，按线程数近线性加速
- **线程安全统计**：所有 `self.stats` 更新均通过 `threading.Lock` 保护，确保并发下数据一致

### 沿用特性
- **智能语义分块**：200~800字分块，保留章节上下文锚点，正确处理HTML表格、LaTeX公式
- **参考文献自动分离**：检测并标记章节末尾参考文献区域，不参与SFT生成
- **双重知识密度过滤**：规则引擎（零API成本）快速预过滤 + LLM精细判断，拦截前言/目录/版权页等低价值内容
- **6种SFT语料类型**：覆盖从基础问答到推理计算的全场景
- **自包含约束检查**：自动过滤含书籍结构指代、图表编号、公式引用等不可独立理解的样本
- **自动化运行报告**：每次运行生成JSON和Markdown双格式报告

---

## 语料类型体系（v3.x）

| 类型标识 | 中文名 | 适合的原文类型 | 典型问题形式 |
|---|---|---|---|
| `fact_qa` | 基础事实问答 | 定义型（含术语、概念、参数说明） | "XXX是什么？"、"XXX的定义是什么？" |
| `comprehension` | 理解型问答 | 过程型（含操作步骤、工艺流程） | "为什么XXX？"、"XXX是如何工作的？" |
| `reasoning_qa` | 推理型问答 | 案例型（含故障分析、因果关系） | 需要"步骤1/步骤2/结论"推理链 |
| `summarization` | 总结任务 | 描述型（含背景介绍、系统描述） | "请总结下面这段描述：..." |
| `information_extraction` | 信息抽取 | 数据型（含参数表、指标对比） | "XXX系统的主要技术参数有哪些？" |
| `calculation` | 计算问题 | 含可推导数学公式的内容 | 含完整LaTeX公式推导过程 |

**内容自动诊断 → 语料类型对应关系：**

```
定义型  →  fact_qa
案例型  →  reasoning_qa
过程型  →  comprehension
描述型  →  summarization
数据型  →  information_extraction
含公式  →  calculation（追加）
```

---

## 项目结构

```
CPT_SFT_CORPUS_PIPELINE/
├── main.py                   # 主控流程 v3.1
├── config.py                 # 全局配置（含 max_workers、新类型分布）
├── src/
│   ├── __init__.py
│   ├── chunker.py            # 智能语义分块器 v2.1
│   ├── cpt_generator.py      # CPT语料生成器（无API调用，稳定）
│   ├── data_loader.py        # Markdown加载与预处理
│   ├── knowledge_filter.py   # 知识密度过滤器 v2.1
│   ├── sft_generator.py      # SFT语料生成器 v3.1 ← 核心优化
│   └── utils.py              # 工具函数
├── readme/
│   ├── README_v3.md          # 详细技术文档
│   └── optimization_analysis.md  # 性能优化分析报告
└── .vscode/
    └── launch.json           # VS Code 调试配置（含4种预设）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai tqdm
```

### 2. 配置 API Key

支持三种方式（优先级从高到低）：

```bash
# 方式一：命令行参数（最直接）
python main.py --api-key sk-xxx ...

# 方式二：环境变量
export DASHSCOPE_API_KEY=sk-xxx   # DashScope（阿里云 Qwen）
export OPENAI_API_KEY=sk-xxx      # OpenAI 兼容接口

# 方式三：config.py 中修改 LLMConfig.api_key 默认值
```

### 3. 基础用法

```bash
# 仅生成CPT语料（不需要API Key）
python main.py --input data/book.md --output output/ --mode cpt

# 仅生成SFT语料
python main.py --input data/book.md --output output/ --mode sft --api-key sk-xxx

# CPT + SFT 同时生成（推荐）
python main.py --input data/book.md --output output/ --mode full --api-key sk-xxx
```

---

## 多线程并发使用指南

### 为什么需要多线程？

v3.x 的SFT生成是**纯I/O密集型任务**：每个chunk向LLM API发送一次请求，然后等待响应。在等待期间CPU完全空闲。单线程模式下，每个chunk严格串行处理，API服务端的并发能力被完全浪费。

```
单线程（workers=1）：
  chunk1 → [等待API响应 ~10s] → chunk2 → [等待API响应 ~10s] → chunk3 ...

多线程（workers=3）：
  chunk1 → [等待API响应 ~10s]
  chunk2 →  [等待API响应 ~10s]          ← 三者同时发出，并行等待
  chunk3 →     [等待API响应 ~10s]
```

### 架构设计

```
process_chunks(chunks, max_workers=N)
       │
       ├─ workers == 1 → _process_sequential()   # 顺序执行，兼容v2.x
       │
       └─ workers  > 1 → _process_parallel()     # 并行执行
                              │
                    ThreadPoolExecutor(max_workers=N)
                              │
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
               _worker()  _worker()  _worker()   # 每个线程独立处理一个chunk
                    │         │         │
               process_chunk() × N（并发）
                    │
               [知识密度过滤] → [合并调用：诊断+生成]
                    │
              as_completed() 收集结果
                    │
              results_lock（保证列表append安全）
                    │
              _stats_lock（保证统计计数安全）
```

**线程安全设计要点：**

| 共享资源 | 保护方式 | 说明 |
|---|---|---|
| `self.stats` 所有字段 | `threading.Lock`（`_stats_lock`） | 通过 `_incr_stat()` / `_incr_type_count()` 原子更新 |
| `all_samples` / `all_diagnoses` 列表 | `threading.Lock`（`results_lock`） | 在 `as_completed()` 回调中统一收集 |
| `OpenAI` 客户端 | 无需锁 | `openai` SDK 的 HTTP 连接池本身是线程安全的 |

### 启用多线程

通过 `--workers` 参数控制并发线程数：

```bash
# 3线程并行（推荐起步配置）
python main.py \
  --input data/book.md \
  --output output/ \
  --mode full \
  --api-key sk-xxx \
  --workers 3

# 5线程（适合高配额账号）
python main.py --input data/book.md --mode sft --api-key sk-xxx --workers 5
```

也可在 `config.py` 中修改默认值：

```python
LLMConfig(
    max_workers=3,   # 修改此处，无需每次传命令行参数
)
```

### workers 数量建议

| API 服务 | 建议 workers | 原因 |
|---|---|---|
| DashScope Qwen（免费额度） | **2~3** | QPM（每分钟请求数）限制较低 |
| DashScope Qwen（付费套餐） | **4~6** | 根据账号QPM上限调整，避免触发限流重试 |
| OpenAI GPT-4系列 | **3~5** | 注意TPM（每分钟Token数）限制 |
| 本地部署（vLLM/Ollama） | **8~16** | 受GPU显存和并发槽位限制 |

> **注意**：`workers` 过大会触发API限流（HTTP 429），导致大量重试反而更慢。建议先用3线程测试，稳定后再按需上调。

### 顺序模式 vs 并行模式对比

| 特性 | 顺序模式（`--workers 1`） | 并行模式（`--workers N`） |
|---|---|---|
| 执行方式 | 每个chunk严格串行，附 `batch_delay=0.5s` 间隔 | N个chunk同时发送API请求 |
| 日志顺序 | 严格按chunk编号顺序输出 | 乱序输出（按完成顺序） |
| 进度回调 | 按顺序精确触发 | 按完成顺序触发，百分比仍准确 |
| 适用场景 | 调试、低配额API账号 | 生产批量处理 |
| 默认值 | `--workers 1` | 需显式指定 |

---

## 完整命令行参数

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` / `-i` | 必填 | 输入MD文件路径或目录路径 |
| `--output` / `-o` | `./output` | 输出目录 |
| `--mode` / `-m` | `full` | `cpt`（仅CPT）/ `sft`（仅SFT）/ `full`（两者） |
| `--config` / `-c` | — | JSON配置文件路径（覆盖所有参数） |

### API 与模型参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--api-key` | — | LLM API Key |
| `--model` | `qwen3-max` | 模型名称 |
| `--base-url` | DashScope URL | API Base URL |
| `--workers` | `1` | **并发线程数**（建议3~5，需符合API限流） |

### 数据处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cold-start` | `0` | 冷启动：仅处理前N个chunk（0=全部） |
| `--min-chars` | `200` | 最小分块字符数 |
| `--max-chars` | `800` | 最大分块字符数 |
| `--domain` | `aerospace` | 领域标识（注入Prompt） |
| `--format` | `sharegpt` | SFT输出格式：`sharegpt` / `alpaca` / `messages` |
| `--glossary` | — | 领域术语表JSON路径 |
| `--log-level` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 过滤参数

| 参数 | 说明 |
|---|---|
| `--no-filter` | 禁用所有知识密度过滤，所有chunk均参与生成 |
| `--no-llm-filter` | 仅用规则引擎过滤，不调用LLM（节省API，速度更快） |

---

## 典型使用场景

### 场景一：首次测试（冷启动验证）

先用少量chunk快速验证流程是否正常，再全量运行：

```bash
python main.py \
  --input data/book.md \
  --output output/ \
  --mode full \
  --api-key sk-xxx \
  --cold-start 30 \
  --workers 3
```

预计耗时：~4分钟（30 chunks）

### 场景二：生产批量处理（推荐配置）

```bash
python main.py \
  --input data/book.md \
  --output output/ \
  --mode full \
  --api-key sk-xxx \
  --workers 4 \
  --no-llm-filter
```

> `--no-llm-filter` 跳过每个"不确定"chunk的LLM过滤判断，进一步减少API调用，适合已验证数据质量的场景。

### 场景三：最大速度（仅规则过滤 + 多线程）

```bash
python main.py \
  --input data/ \
  --output output/ \
  --mode sft \
  --api-key sk-xxx \
  --workers 5 \
  --no-llm-filter \
  --domain aerospace
```

### 场景四：VS Code 调试

直接按 `F5`，从以下预设配置中选择：

- **`full模式冷启动30chunk（v3.1测试）`** — 日常验证首选
- **`SFT模式（微小卫星，完整运行）`** — 全量生产
- **`CPT模式（无需API）`** — 纯CPT语料，免费
- **`禁用LLM过滤+3线程（快速批量）`** — 最快速度

---

## Pipeline 执行流程

```
MD文件输入
    │
    ▼ Step 1
[文档加载] MarkdownLoader
    解析章节结构、标题层级、自动标记参考文献区域
    │
    ▼ Step 2
[语义分块] SemanticChunker v2.1
    200~800字分块，保留章节上下文锚点
    特殊处理：HTML表格整行原子化、LaTeX公式块不拆分
    │
    ▼ Step 3
[CPT语料生成] CPTGenerator（无API调用）
    基础模式：chunk内容 + 元数据头
    上下文模式：滑窗合并同章节相邻chunk
    │
    ▼ Step 4
[知识密度预过滤] KnowledgeDensityFilter v2.1
    第一层：规则引擎（零API成本）
      → 直接过滤：过短文本、前言/目录/版权页标题、参考文献格式内容
      → 直接通过：命中≥2个高价值知识模式（概念定义/技术原理/系统结构等）
    第二层：LLM精细判断（仅对规则引擎"不确定"的内容，可用--no-llm-filter跳过）
    │
    ▼ Step 5  ← v3.1 核心优化
[SFT语料生成] SFTGenerator v3.1

    ┌─ 单线程（workers=1）─────────────────────────────────────┐
    │  for chunk in filtered_chunks:                          │
    │      filter → 合并调用(诊断+生成) → 收集样本             │
    │      sleep(batch_delay=0.5s)                           │
    └─────────────────────────────────────────────────────────┘

    ┌─ 多线程（workers=N）─────────────────────────────────────┐
    │  ThreadPoolExecutor(max_workers=N)                      │
    │  ├── Thread-1: filter → 合并调用 → 返回结果              │
    │  ├── Thread-2: filter → 合并调用 → 返回结果  （并发）    │
    │  └── Thread-N: filter → 合并调用 → 返回结果              │
    │  as_completed() 收集 → results_lock 合并列表             │
    └─────────────────────────────────────────────────────────┘

    合并调用（SYSTEM_PROMPT_MERGED）单次返回：
    {
      "diagnosis": "定义型",
      "confidence": 0.88,
      "selected_types": ["fact_qa", "information_extraction"],
      "samples": [
        {"type": "fact_qa", "question": "...", "answer": "..."},
        {"type": "information_extraction", "question": "...", "answer": "..."}
      ]
    }
    │
    ▼ Step 6~7
[报告生成] 输出 pipeline_report.md + pipeline_report.json
```

---

## 输出文件说明

每次运行在 `output/<时间戳>/` 下生成：

```
output/20260312_225318/
├── pipeline.log                      # 完整运行日志
├── config_snapshot.json              # 本次配置快照（可复现）
├── chunks.jsonl                      # 所有分块结果（含type_hint、has_formula等）
├── cpt_<书名>.jsonl                  # CPT语料（基础模式）
├── cpt_contextual_<书名>.jsonl       # CPT语料（上下文滑窗模式）
├── filter_details.jsonl              # 每个chunk的过滤决策详情
├── filtered_chunks.jsonl             # 被过滤的低价值chunk（供审查）
├── diagnoses.jsonl                   # 通过过滤的chunk的诊断+生成信息
├── sft_train.sharegpt.jsonl          # SFT语料，ShareGPT格式（主训练集）
├── sft_train.alpaca.jsonl            # SFT语料，Alpaca格式（同步输出）
├── sft_review_pool.jsonl             # 低置信度样本（人工审核池）
├── pipeline_report.json              # JSON格式运行报告
└── pipeline_report.md                # Markdown格式可读报告
```

**ShareGPT格式样本示例：**

```json
{
  "type": "fact_qa",
  "source_chunk_id": "微小卫星_ch第1章_sec微小卫星技术_p0015",
  "conversations": [
    {"from": "user", "value": "微小卫星姿态确定系统的主要功能是什么？"},
    {"from": "assistant", "value": "微小卫星姿态确定系统的主要功能是..."}
  ],
  "confidence": 0.88,
  "diagnosis": "定义型",
  "metadata": {
    "chapter": "第1章 绪论",
    "section": "1.2 微小卫星技术发展",
    "context_anchor": "第1章 绪论 > 1.2 微小卫星技术发展",
    "knowledge_density": "high",
    "matched_categories": ["概念定义", "系统结构"]
  }
}
```

---

## 性能基准

以《微小卫星总体设计与工程实践》（200页，约677个chunk，有效chunk约450个）为参考：

| 配置 | 预计耗时 | API调用次数 | 说明 |
|---|---|---|---|
| v2.x 单线程 | **5~7 小时** | ~1000次 | 每chunk 2~3次调用 |
| v3.1 单线程（`--workers 1`） | **2~3 小时** | ~500次 | 合并调用，减少50% |
| v3.1 三线程（`--workers 3`） | **40~70 分钟** | ~500次 | 并发3倍加速 |
| v3.1 五线程（`--workers 5`） | **25~45 分钟** | ~500次 | 并发5倍加速 |

> 实测数据（30 chunk冷启动，workers=3）：**227秒，89条SFT样本，24次API调用**

---

## 更多文档

- [`readme/optimization_analysis.md`](readme/optimization_analysis.md)：v3.1性能优化的完整分析报告，包含瓶颈定位、方案对比和数据估算
- [`readme/README_v3.md`](readme/README_v3.md)：详细技术架构文档
