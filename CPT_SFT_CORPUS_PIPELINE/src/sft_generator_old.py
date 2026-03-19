"""
SFT（Supervised Fine-Tuning）语料动态生成器 v2.2
=================================================
融合v1的完整语料类型体系与v2的知识密度过滤机制。

核心设计：
1. 前置关卡：知识密度过滤（规则引擎 + LLM双重过滤），拦截低价值内容
2. 阶段1（内容诊断）：分析通过过滤的chunk的内容属性，判断最适合的语料类型
3. 阶段2（定向生成）：根据诊断结果，使用对应的Prompt模板生成语料

支持的语料类型（完整6种）：
1. knowledge_qa       - 知识点问答（适合定义型/描述型/数据型内容）
2. concept_explanation - 核心概念阐述（适合定义型内容）
3. text_summary       - 文本摘要（适合描述型/数据型内容）
4. causal_reasoning   - 因果推理（适合案例型内容）
5. cot_reasoning      - 思维链推理（适合案例型/过程型内容）
6. process_qa         - 流程步骤问答（适合过程型内容）

内容诊断类型（5种）：
- 定义型（含术语定义、概念界定、参数说明）
- 案例型（含故障分析、事故案例、问题-原因-结论）
- 过程型（含操作步骤、工艺流程、测试程序）
- 描述型（事实陈述、背景介绍、系统描述）
- 数据型（含数据表格、参数对比、统计信息） 
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .chunker import Chunk
from .knowledge_filter import KnowledgeDensityFilter, FilterResult
from .utils import (
    setup_logger, safe_json_parse, get_timestamp,
    save_jsonl, count_chars
)

logger = setup_logger(__name__)


# ============================================================
# Prompt模板定义
# ============================================================

SYSTEM_PROMPT_DIAGNOSIS = """你是一名专业的{domain}领域教材语料工程师。你的任务是分析给定文本的内容属性，为后续的微调语料生成提供精确的类型判断。

请严格按以下流程处理：

【步骤1：内容诊断】
分析文本的核心属性（单选）：
□ 定义型（含术语定义、概念界定、参数说明、分类标准）
□ 案例型（含故障分析、事故案例、问题-原因-结论、工程实例）
□ 过程型（含操作步骤、工艺流程、测试程序、设计方法）
□ 描述型（事实陈述、背景介绍、系统描述、发展历史）
□ 数据型（含数据表格、参数对比、统计信息、指标约束）

注意：请仔细区分，不要将所有内容都归为"描述型"。判断标准：
- 如果文本包含"是指"、"定义为"、"称为"、"分为"等定义性表述 → 定义型
- 如果文本包含故障、异常、原因分析、解决方案等 → 案例型
- 如果文本包含步骤、流程、程序、方法等操作性内容 → 过程型
- 如果文本包含具体数值、参数表、指标对比等 → 数据型
- 仅当文本主要是背景介绍、历史叙述、一般性描述时 → 描述型

【步骤2：判断依据】
简要说明你的判断理由（1-2句话）。

【步骤3：语料类型选择】
根据诊断结果，从以下6种语料类型中选择2-3种最适合的类型：
- knowledge_qa: 知识点问答（适合任何有知识含量的内容）
- concept_explanation: 核心概念阐述（适合有明确概念/术语的内容）
- text_summary: 文本摘要（适合信息密集的段落）
- causal_reasoning: 因果推理（适合有因果关系的内容）
- cot_reasoning: 思维链推理（适合需要多步分析的内容）
- process_qa: 流程步骤问答（适合有操作流程的内容）

选择原则：
1. 每个文本至少选择2种不同类型，增加语料多样性
2. knowledge_qa是通用类型，几乎所有高价值文本都适合
3. 如果文本中有任何因果关系（"因为"、"导致"、"由于"），应包含causal_reasoning
4. 如果文本中有系统组成或技术原理，应包含cot_reasoning
5. 如果文本中有明确的概念定义，应包含concept_explanation

【输出格式】
必须输出纯JSON，格式如下：
{{"diagnosis": "定义型/案例型/过程型/描述型/数据型", "confidence": 0.0-1.0, "reason": "判断理由", "sub_types": ["语料类型1", "语料类型2", "语料类型3"]}}"""


SYSTEM_PROMPT_GENERATION = """你是一名专业的{domain}领域教材语料工程师。请根据给定的原文内容，严格按照指定的语料类型生成高质量的微调训练样本。

本任务目标是构建高质量领域知识语料库，用于：
- 领域大模型训练
- 知识库构建
- 向量数据库检索
- 专业问答系统

【硬性约束 - 必须严格遵守】
1. 所有内容必须严格源自原文，禁止编造任何事实、数据或结论
2. 推理步骤需标注"步骤1/步骤2/步骤3"等序号
3. 问题的设计应当自然、专业，像真实用户会提出的问题
4. 回答应当完整、准确、专业，适合作为{domain}领域的训练数据
5. 避免生成常识问题，优先生成技术型问题
6. 必须输出纯JSON格式

【自包含约束 - 严禁违反】
问句和答案必须是完全自包含的，即脱离原文后仍然可以独立理解。
严禁在问句或答案中出现以下内容：
- 书籍结构指代："本书"、"前言"、"第X章"、"X.X节"、"X.X.X节"、"本章"、"本节"、"上一节"、"下一节"
- 模糊指代词："文中提到"、"上文所述"、"如前所述"、"根据上述内容"、"上面提到的"
- 图表引用："如图X所示"、"如表X所示"、"见图X"、"参见表X"、"图X表明"
- 公式编号引用："公式(X.X)"、"由式(X)可得"、"代入公式X"、"根据公式X"

如果原文中涉及公式，必须输出完整的 LaTeX 数学表达式，而非引用编号。
错误示例："根据公式(5.1)可以计算推进剂质量"
正确示例："根据齐奥尔科夫斯基公式 $\\Delta v = I_{{sp}} g_0 \\ln(m_0/m_f)$ 可以计算推进剂质量"

错误示例："请概括本书前言及第5章的核心内容"
正确示例："微小卫星总体参数预算的主要内容包括哪些方面？"

错误示例："如表5.1所示，卫星各分系统的质量分配是怎样的？"
正确示例："微小卫星各分系统的质量分配通常是怎样的？"

{negative_example}

{glossary_context}"""


# 各类型的生成Prompt模板（完整6种）
# 注意：所有模板都遵循系统Prompt中的自包含约束和去指代约束
GENERATION_PROMPTS = {
    "knowledge_qa": """请基于以下原文，生成{n}组高质量的知识问答对。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 问题应覆盖原文的核心知识点
- 问题类型多样化：包括"是什么"、"为什么"、"如何"、"有哪些"等
- 回答应完整准确，严格基于原文
- 适当使用专业术语
- 避免生成常识问题，优先生成技术型问题
- 问题必须自包含，不得引用章节号、图表号或公式编号
- 回答中如涉及公式，必须写出完整的LaTeX表达式（如 $F=ma$），不得写"公式(X.X)"
- 回答中如涉及表格数据，必须直接给出具体数值，不得写"如表X所示"

【正确问句示例】
- "微小卫星质量预算的主要目的是什么？"
- "太阳电池阵发电总功率的计算需要考虑哪些因素？"
- "卫星姿态确定系统的精度指标通常如何分配？"

【错误问句示例 - 严禁出现】
- "请概括第5章关于总体参数预算的内容"（引用了章节号）
- "表5.1中各分系统的质量分配是怎样的？"（引用了表号）
- "根据上述内容，卫星质量预算有哪些要求？"（使用了指代词）

输出JSON格式：
{{"type": "knowledge_qa", "samples": [{{"question": "问题", "answer": "回答"}}, ...]}}""",

    "concept_explanation": """请基于以下原文，生成{n}组核心概念阐述样本。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 提取原文中的核心概念/术语
- 对每个概念给出：概念名称、简明定义、详细阐述
- 阐述应包含概念的本质特征、应用场景或重要性
- 严格基于原文，不得编造
- 问句必须自包含，直接提问概念本身，不得引用章节或段落
- 如涉及公式，必须在答案中写出完整的LaTeX表达式

【正确问句示例】
- "什么是卫星的比功率？请解释其含义和工程意义。"
- "请解释微小卫星中'单机质量预算'的概念及其作用。"

【错误问句示例 - 严禁出现】
- "请解释5.1.1节中提到的质量预算概念"（引用了节号）
- "文中提到的'比功率'是什么意思？"（使用了'文中提到'）

输出JSON格式：
{{"type": "concept_explanation", "samples": [{{"concept": "概念名称", "question": "什么是XXX？/请解释XXX的含义", "answer": "概念：XXX\\n定义：...\\n阐述：..."}}, ...]}}""",

    "text_summary": """请基于以下原文，生成{n}组文本摘要样本。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 摘要应保留原文的关键信息和核心观点
- 包含不同粒度的摘要：一句话摘要、段落摘要
- 问题形式可以是"请总结..."、"概括..."、"简述..."
- 严格基于原文，不得遗漏关键信息
- 问句必须用具体的技术主题来提问，不得引用章节号或使用"本书"、"本章"等指代
- 如涉及公式或数据，答案中必须给出完整表达式或具体数值

【正确问句示例】
- "请简述微小卫星质量预算的基本原则和方法。"
- "请概括卫星功耗预算中太阳电池阵发电功率的计算要点。"

【错误问句示例 - 严禁出现】
- "请用一句话概括本书前言及第5章的核心内容。"（引用了'本书前言'和章节号）
- "请概括5.1.2节关于功耗预算的核心内容。"（引用了节号）
- "请总结上述内容的要点。"（使用了'上述内容'）

输出JSON格式：
{{"type": "text_summary", "samples": [{{"question": "请总结/概括...", "answer": "摘要内容"}}, ...]}}""",

    "causal_reasoning": """请基于以下原文，生成{n}组因果推理样本。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 问题应聚焦于原文中的因果关系
- 回答应包含清晰的因果链：原因→机理→结果
- 适合的问题形式："为什么..."、"...的原因是什么"、"...导致了什么后果"
- 如果涉及故障/事故，应包含：现象描述→原因分析→结论
- 严格基于原文的因果逻辑
- 问句和答案必须自包含，不得引用章节号、图表号或公式编号
- 如涉及公式推导，必须写出完整的LaTeX数学表达式

输出JSON格式：
{{"type": "causal_reasoning", "samples": [{{"question": "因果推理问题", "answer": "步骤1：...\\n步骤2：...\\n结论：..."}}, ...]}}""",

    "cot_reasoning": """请基于以下原文，生成{n}组思维链(Chain-of-Thought)推理样本。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 问题应需要多步推理才能回答
- 回答必须展示完整的推理过程，使用"步骤1→步骤2→步骤3"的格式
- 每个步骤应包含：推理依据 + 推理过程 + 阶段结论
- 最终给出明确的总结论
- 适合的问题形式："请分析..."、"请推理..."、"如果...那么..."
- 严格基于原文进行推理，不得引入原文未提及的信息
- 问句和答案必须自包含，不得引用章节号、图表号或公式编号
- 推理过程中如涉及公式，必须写出完整的LaTeX表达式（如 $\\Delta v = I_{{sp}} g_0 \\ln(m_0/m_f)$），不得写"由公式(X.X)可得"
- 推理依据应直接引述技术事实，不得写"原文指出"、"文中提到"等指代

【正确推理依据示例】
- "依据：卫星总质量受运载火箭运载能力约束，通常需要留出10%-15%的质量余量。"

【错误推理依据示例 - 严禁出现】
- "依据：原文指出卫星总质量受运载约束"（使用了'原文指出'）
- "依据：由公式(5.3)可知..."（引用了公式编号）

输出JSON格式：
{{"type": "cot_reasoning", "samples": [{{"question": "需要多步推理的问题", "answer": "让我们逐步分析这个问题。\\n\\n步骤1：...\\n依据：...\\n\\n步骤2：...\\n依据：...\\n\\n步骤3：...\\n\\n综合以上分析，结论是：..."}}, ...]}}""",

    "process_qa": """请基于以下原文，生成{n}组流程/步骤问答样本。

【原文内容】
{context_anchor}
{content}

【前文摘要】{prev_summary}

【要求】
- 问题应聚焦于原文中的操作流程、工艺步骤或测试程序
- 回答应清晰列出步骤序号和每步的具体操作
- 适合的问题形式："...的操作步骤是什么"、"如何进行..."、"...的流程是怎样的"
- 严格基于原文的流程描述
- 问句必须自包含，直接描述技术流程主题，不得引用章节号
- 如涉及公式或参数，必须在答案中给出完整表达式或具体数值

输出JSON格式：
{{"type": "process_qa", "samples": [{{"question": "流程/步骤问题", "answer": "步骤1：...\\n步骤2：...\\n步骤3：..."}}, ...]}}"""
}


# 类型名称标准化映射（修复LLM返回非标准类型名的问题）
TYPE_NORMALIZE = {
    # 中文类型名 → 标准英文名
    "知识问答": "knowledge_qa",
    "知识_问答": "knowledge_qa",
    "知识点问答": "knowledge_qa",
    "概念阐述": "concept_explanation",
    "概念_阐述": "concept_explanation",
    "核心概念阐述": "concept_explanation",
    "文本摘要": "text_summary",
    "文本_摘要": "text_summary",
    "摘要": "text_summary",
    "因果推理": "causal_reasoning",
    "因果_推理": "causal_reasoning",
    "思维链推理": "cot_reasoning",
    "思维链_推理": "cot_reasoning",
    "cot推理": "cot_reasoning",
    "流程问答": "process_qa",
    "流程_问答": "process_qa",
    "步骤问答": "process_qa",
    "过程_流程": "process_qa",
    # 非标准英文名 → 标准英文名
    "data_type": "knowledge_qa",
    "qa": "knowledge_qa",
    "summary": "text_summary",
    "explanation": "concept_explanation",
    "reasoning": "cot_reasoning",
    "causal": "causal_reasoning",
    "process": "process_qa",
}

# 标准语料类型集合
VALID_TYPES = {"knowledge_qa", "concept_explanation", "text_summary", 
               "causal_reasoning", "cot_reasoning", "process_qa"}


def normalize_type(type_name: str) -> Optional[str]:
    """将LLM返回的类型名称标准化为合法的语料类型"""
    if type_name in VALID_TYPES:
        return type_name
    normalized = TYPE_NORMALIZE.get(type_name)
    if normalized:
        return normalized
    # 模糊匹配：检查是否包含关键词
    type_lower = type_name.lower()
    for key, val in TYPE_NORMALIZE.items():
        if key in type_lower or type_lower in key:
            return val
    return None


# 内容类型到语料类型的映射（完整映射表）
TYPE_MAPPING = {
    # 中文诊断标签 → 语料类型
    "定义型": ["knowledge_qa", "concept_explanation"],
    "案例型": ["causal_reasoning", "cot_reasoning"],
    "过程型": ["process_qa", "cot_reasoning"],
    "描述型": ["text_summary", "knowledge_qa"],
    "数据型": ["knowledge_qa", "text_summary"],
    # 英文兼容
    "definition": ["knowledge_qa", "concept_explanation"],
    "case": ["causal_reasoning", "cot_reasoning"],
    "process": ["process_qa", "cot_reasoning"],
    "description": ["text_summary", "knowledge_qa"],
    "data": ["knowledge_qa", "text_summary"],
    # v2知识密度过滤器的知识子类别 → 语料类型（兼容映射）
    "概念定义": ["knowledge_qa", "concept_explanation"],
    "技术原理": ["cot_reasoning", "knowledge_qa"],
    "系统结构": ["knowledge_qa", "concept_explanation"],
    "工程流程": ["process_qa", "cot_reasoning"],
    "技术参数": ["knowledge_qa", "text_summary"],
    "案例分析": ["causal_reasoning", "cot_reasoning"],
    "技术对比": ["knowledge_qa", "text_summary"],
    # 知识密度过滤器的诊断标签
    "知识型": ["knowledge_qa", "concept_explanation"],
}


# 负面示例模板
NEGATIVE_EXAMPLE = """【负面示例 - 请避免】
错误做法：编造原文未提及的解决方案或数据
正确做法：仅基于原文内容进行分析和回答

错误示例：
问：某型号发动机推力不足的原因？
答：可能是燃料纯度不够（❌ 原文未提及燃料纯度问题）

正确示例：
问：某型号发动机推力不足的原因？
答：根据原文分析，推力不足的主要原因是涡轮泵转速下降，导致燃料供给流量不足。（✅ 严格基于原文）"""


@dataclass
class SFTSample:
    """SFT训练样本"""
    type: str                           # 语料类型（6种之一）
    source_chunk_id: str                # 来源chunk ID
    conversations: List[Dict]           # 对话列表 [{"from": "user/assistant", "value": "..."}]
    confidence: float = 0.0             # 置信度
    diagnosis: str = ""                 # 内容诊断结果
    metadata: Dict = field(default_factory=dict)
    # metadata包含:
    #   chapter, section, context_anchor, diagnosis_reason,
    #   char_count_q, char_count_a,
    #   knowledge_density (来自过滤器), matched_categories (来自过滤器),
    #   filter_stage (来自过滤器)


class SFTGenerator:
    """
    SFT语料动态生成器 v2.2
    
    融合v1完整类型体系 + v2知识密度过滤的三阶段策略：
    0. 前置关卡：知识密度过滤（规则引擎 + LLM双重过滤），拦截低价值内容
    1. 阶段1（内容诊断）：调用LLM判断chunk的内容类型（5种）
    2. 阶段2（定向生成）：根据诊断结果，使用对应的Prompt模板生成语料（6种）
    
    支持的LLM API：
    - DashScope（Qwen3-Max）：通过OpenAI兼容接口调用
    - OpenAI兼容接口：支持任何兼容OpenAI API的服务
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-max",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        batch_delay: float = 1.0,
        domain: str = "aerospace",
        confidence_threshold: float = 0.7,
        max_samples_per_chunk: int = 5,
        enable_negative_examples: bool = True,
        enable_thinking: bool = False,
        enable_llm_filter: bool = True,
        glossary: Dict[str, str] = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_delay = batch_delay
        self.domain = domain
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples_per_chunk
        self.enable_negative = enable_negative_examples
        self.enable_thinking = enable_thinking
        self.glossary = glossary or {}
        
        # 初始化OpenAI客户端
        self.client = None
        self._init_client()
        
        # 初始化知识密度过滤器
        self.knowledge_filter = KnowledgeDensityFilter(
            min_chars=30,
            high_value_threshold=2,
            enable_llm_filter=enable_llm_filter,
            domain=domain
        )
        
        # 统计信息
        self.stats = {
            "total_chunks_input": 0,
            "chunks_filtered": 0,
            "chunks_passed": 0,
            "total_api_calls": 0,
            "total_tokens_used": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "low_confidence_count": 0,
            "type_counts": {},
            "filter_stats": {}
        }
        
        logger.info(
            f"SFT生成器v2.2初始化: model={model}, domain={domain}, "
            f"temperature={temperature}, llm_filter={enable_llm_filter}"
        )
    
    def _init_client(self):
        """初始化OpenAI兼容客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"API客户端初始化成功: {self.base_url}")
        except ImportError:
            logger.error("请安装openai库: pip install openai")
            raise
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        调用LLM API（带重试机制）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            str: LLM响应文本，失败返回None
        """
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                }
                
                # Qwen3思考模式支持
                if self.enable_thinking:
                    kwargs["extra_body"] = {"enable_thinking": True}
                
                response = self.client.chat.completions.create(**kwargs)
                
                self.stats["total_api_calls"] += 1
                if response.usage:
                    self.stats["total_tokens_used"] += response.usage.total_tokens
                
                content = response.choices[0].message.content
                return content
                
            except Exception as e:
                logger.warning(
                    f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        self.stats["failed_generations"] += 1
        return None
    
    def diagnose_chunk(self, chunk: Chunk, filter_result: FilterResult = None) -> Optional[Dict]:
        """
        阶段1：内容诊断
        
        调用LLM分析chunk的内容类型，返回诊断结果。
        如果知识密度过滤器已提供了匹配的知识类别，可作为辅助参考。
        
        Args:
            chunk: 待诊断的chunk
            filter_result: 知识密度过滤结果（可选）
            
        Returns:
            Dict: {"diagnosis": "类型", "confidence": 0.0-1.0, "sub_types": [...]}
        """
        system_prompt = SYSTEM_PROMPT_DIAGNOSIS.format(domain=self.domain)
        
        # 构建用户Prompt，如果有过滤器结果则附加参考信息
        filter_hint = ""
        if filter_result and filter_result.matched_categories:
            filter_hint = f"\n【预分析参考】该文本被初步识别为包含以下知识类别：{', '.join(filter_result.matched_categories)}。请综合判断。"
        
        user_prompt = f"""请分析以下文本的内容属性：

【章节位置】{chunk.context_anchor}
【文本内容】
{chunk.content}{filter_hint}"""
        
        response = self._call_llm(system_prompt, user_prompt)
        
        if not response:
            # API失败时，使用规则引擎的type_hint作为fallback
            logger.warning(f"诊断API失败，使用规则引擎结果: {chunk.type_hint}")
            fallback_diagnosis = chunk.type_hint or "描述型"
            return {
                "diagnosis": fallback_diagnosis,
                "confidence": 0.5,
                "reason": "规则引擎自动判断（API不可用）",
                "sub_types": TYPE_MAPPING.get(fallback_diagnosis, ["knowledge_qa"])
            }
        
        result = safe_json_parse(response)
        
        if result and "diagnosis" in result:
            # 对LLM返回的sub_types进行标准化处理
            if "sub_types" in result and result["sub_types"]:
                normalized = []
                for t in result["sub_types"]:
                    nt = normalize_type(t)
                    if nt:
                        normalized.append(nt)
                    else:
                        logger.warning(f"无法识别的语料类型 '{t}'，已跳过")
                result["sub_types"] = list(dict.fromkeys(normalized))  # 去重保序
            
            # 确保sub_types字段存在且非空
            if "sub_types" not in result or not result["sub_types"]:
                diagnosis = result["diagnosis"]
                result["sub_types"] = TYPE_MAPPING.get(
                    diagnosis, ["knowledge_qa"]
                )
            
            # 智能增强：利用过滤器的知识类别信息丰富语料类型
            if filter_result and filter_result.matched_categories:
                extra_types = set()
                for cat in filter_result.matched_categories:
                    cat_types = TYPE_MAPPING.get(cat, [])
                    extra_types.update(cat_types)
                # 将过滤器识别的额外类型添加到sub_types中（去重）
                current_types = set(result["sub_types"])
                new_types = extra_types - current_types
                if new_types:
                    result["sub_types"].extend(list(new_types)[:2])  # 最多追加2种
                    logger.debug(
                        f"智能增强: 基于过滤器类别{filter_result.matched_categories}"
                        f"追加语料类型: {new_types}"
                    )
            
            # 确保至少有2种类型，增加多样性
            if len(result["sub_types"]) < 2:
                if "knowledge_qa" not in result["sub_types"]:
                    result["sub_types"].append("knowledge_qa")
                elif "text_summary" not in result["sub_types"]:
                    result["sub_types"].append("text_summary")
            
            return result
        else:
            logger.warning(f"诊断结果解析失败，使用规则引擎: {response[:200]}")
            fallback_diagnosis = chunk.type_hint or "描述型"
            return {
                "diagnosis": fallback_diagnosis,
                "confidence": 0.5,
                "reason": "JSON解析失败，使用规则引擎",
                "sub_types": TYPE_MAPPING.get(fallback_diagnosis, ["knowledge_qa"])
            }
    
    def generate_samples(
        self, 
        chunk: Chunk, 
        diagnosis: Dict,
        filter_result: FilterResult = None
    ) -> List[SFTSample]:
        """
        阶段2：定向生成
        
        根据诊断结果，使用对应的Prompt模板生成SFT样本。
        
        Args:
            chunk: 源chunk
            diagnosis: 诊断结果
            filter_result: 知识密度过滤结果（可选，用于元数据）
            
        Returns:
            List[SFTSample]: 生成的SFT样本列表
        """
        sub_types = diagnosis.get("sub_types", ["knowledge_qa"])
        confidence = diagnosis.get("confidence", 0.5)
        
        all_samples = []
        
        # 构建通用上下文
        negative_example = NEGATIVE_EXAMPLE if self.enable_negative else ""
        glossary_context = self._build_glossary_context(chunk.content)
        
        system_prompt = SYSTEM_PROMPT_GENERATION.format(
            domain=self.domain,
            negative_example=negative_example,
            glossary_context=glossary_context
        )
        
        for stype in sub_types:
            if stype not in GENERATION_PROMPTS:
                logger.warning(f"未知的语料类型: {stype}")
                continue
            
            # 计算每种类型生成的样本数
            n_samples = max(1, self.max_samples // len(sub_types))
            
            # 构建生成Prompt
            prompt_template = GENERATION_PROMPTS[stype]
            user_prompt = prompt_template.format(
                n=n_samples,
                content=chunk.content,
                context_anchor=chunk.context_anchor,
                prev_summary=chunk.prev_summary or "（无前文）"
            )
            
            # 调用LLM生成
            response = self._call_llm(system_prompt, user_prompt)
            
            if not response:
                logger.warning(f"生成失败: chunk={chunk.chunk_id}, type={stype}")
                continue
            
            # 解析生成结果
            parsed = safe_json_parse(response)
            
            if parsed and "samples" in parsed:
                for sample_data in parsed["samples"]:
                    question = sample_data.get("question", "")
                    answer = sample_data.get("answer", "")
                    
                    if not question or not answer:
                        continue
                    
                    # 质量检查：答案长度
                    if len(answer.strip()) < 10:
                        logger.debug(f"答案过短，跳过: {answer[:50]}")
                        continue
                    
                    # 质量检查：自包含约束验证
                    violation = self._check_self_contained(question, answer)
                    if violation:
                        logger.warning(
                            f"自包含约束违反，跳过: [{violation}] "
                            f"Q={question[:60]}..."
                        )
                        self.stats["self_contained_violations"] = \
                            self.stats.get("self_contained_violations", 0) + 1
                        continue
                    
                    # 构建ShareGPT格式的对话
                    conversations = [
                        {"from": "user", "value": question},
                        {"from": "assistant", "value": answer}
                    ]
                    
                    # 构建元数据
                    metadata = {
                        "chapter": chunk.chapter_title,
                        "section": chunk.section_title,
                        "context_anchor": chunk.context_anchor,
                        "diagnosis_reason": diagnosis.get("reason", ""),
                        "char_count_q": len(question),
                        "char_count_a": len(answer)
                    }
                    
                    # 如果有过滤器结果，附加过滤相关元数据
                    if filter_result:
                        metadata.update({
                            "knowledge_density": filter_result.knowledge_density,
                            "matched_categories": filter_result.matched_categories,
                            "filter_stage": filter_result.filter_stage
                        })
                    
                    sample = SFTSample(
                        type=stype,
                        source_chunk_id=chunk.chunk_id,
                        conversations=conversations,
                        confidence=confidence,
                        diagnosis=diagnosis.get("diagnosis", ""),
                        metadata=metadata
                    )
                    all_samples.append(sample)
                    
                    # 更新统计
                    self.stats["successful_generations"] += 1
                    self.stats["type_counts"][stype] = \
                        self.stats["type_counts"].get(stype, 0) + 1
            else:
                logger.warning(
                    f"生成结果解析失败: chunk={chunk.chunk_id}, type={stype}, "
                    f"response={response[:200] if response else 'None'}"
                )
                self.stats["failed_generations"] += 1
        
        return all_samples
    
    def process_chunk(self, chunk: Chunk) -> Tuple[List[SFTSample], Dict]:
        """
        处理单个chunk的完整流程：过滤 → 诊断 → 生成
        
        Returns:
            Tuple[List[SFTSample], Dict]: (生成的样本列表, 诊断+过滤结果)
        """
        logger.info(
            f"处理chunk: {chunk.chunk_id} "
            f"(type_hint={chunk.type_hint}, chars={chunk.char_count})"
        )
        
        # ========== 前置关卡：知识密度过滤 ==========
        filter_result = self.knowledge_filter.filter_chunk(
            chunk, 
            llm_callback=self._call_llm
        )
        
        diagnosis_info = {
            "chunk_id": chunk.chunk_id,
            "knowledge_density": filter_result.knowledge_density,
            "filter_diagnosis": filter_result.diagnosis,
            "filter_reason": filter_result.reason,
            "should_generate": filter_result.should_generate,
            "filter_stage": filter_result.filter_stage,
            "filter_confidence": filter_result.confidence,
            "matched_categories": filter_result.matched_categories
        }
        
        # 低价值内容直接跳过
        if not filter_result.should_generate:
            logger.info(
                f"[过滤] {chunk.chunk_id} | "
                f"密度={filter_result.knowledge_density} | "
                f"原因={filter_result.reason}"
            )
            self.stats["chunks_filtered"] += 1
            return [], diagnosis_info
        
        self.stats["chunks_passed"] += 1
        
        logger.info(
            f"[通过] {chunk.chunk_id} | "
            f"密度={filter_result.knowledge_density} | "
            f"类别={filter_result.matched_categories}"
        )
        
        # ========== 阶段1：内容诊断 ==========
        diagnosis = self.diagnose_chunk(chunk, filter_result)
        
        if not diagnosis:
            logger.error(f"诊断失败: {chunk.chunk_id}")
            return [], diagnosis_info
        
        logger.info(
            f"诊断结果: {diagnosis['diagnosis']} "
            f"(confidence={diagnosis.get('confidence', 0):.2f}, "
            f"sub_types={diagnosis.get('sub_types', [])})"
        )
        
        # 更新诊断信息
        diagnosis_info.update({
            "content_diagnosis": diagnosis.get("diagnosis", ""),
            "content_confidence": diagnosis.get("confidence", 0),
            "content_reason": diagnosis.get("reason", ""),
            "sub_types": diagnosis.get("sub_types", [])
        })
        
        # 低置信度标记
        if diagnosis.get("confidence", 0) < self.confidence_threshold:
            self.stats["low_confidence_count"] += 1
            logger.warning(
                f"低置信度chunk: {chunk.chunk_id} "
                f"(confidence={diagnosis.get('confidence', 0):.2f})"
            )
        
        # ========== 阶段2：定向生成 ==========
        samples = self.generate_samples(chunk, diagnosis, filter_result)
        
        logger.info(f"生成完成: {len(samples)} 条样本")
        
        return samples, diagnosis_info
    
    def process_chunks(
        self, 
        chunks: List[Chunk],
        progress_callback=None
    ) -> Tuple[List[SFTSample], List[SFTSample], List[Dict]]:
        """
        批量处理chunk列表
        
        Args:
            chunks: chunk列表
            progress_callback: 进度回调函数 callback(current, total, chunk_id)
            
        Returns:
            Tuple: (合格样本列表, 低置信度样本列表, 诊断结果列表)
        """
        self.stats["total_chunks_input"] = len(chunks)
        logger.info(f"开始批量处理 {len(chunks)} 个chunk...")
        
        all_samples = []
        low_confidence_samples = []
        all_diagnoses = []
        
        for i, chunk in enumerate(chunks):
            # 进度回调
            if progress_callback:
                progress_callback(i + 1, len(chunks), chunk.chunk_id)
            
            try:
                samples, diagnosis = self.process_chunk(chunk)
                all_diagnoses.append(diagnosis)
                
                # 按置信度分流
                for s in samples:
                    if s.confidence >= self.confidence_threshold:
                        all_samples.append(s)
                    else:
                        low_confidence_samples.append(s)
                
            except Exception as e:
                logger.error(f"处理chunk失败 {chunk.chunk_id}: {e}")
                self.stats["failed_generations"] += 1
            
            # 批次间延迟（避免API限流）
            if i < len(chunks) - 1:
                time.sleep(self.batch_delay)
        
        # 更新过滤统计
        self.stats["filter_stats"] = self.knowledge_filter.get_statistics()
        
        logger.info(
            f"批量处理完成: "
            f"输入={len(chunks)}, "
            f"过滤={self.stats['chunks_filtered']}, "
            f"通过={self.stats['chunks_passed']}, "
            f"合格样本={len(all_samples)}, "
            f"低置信度样本={len(low_confidence_samples)}, "
            f"总API调用={self.stats['total_api_calls']}"
        )
        
        return all_samples, low_confidence_samples, all_diagnoses
    
    def _check_self_contained(self, question: str, answer: str) -> Optional[str]:
        """
        检查问句和答案是否违反自包含约束。
        
        Returns:
            违反原因字符串，如果没有违反则返回None
        """
        combined = question + " " + answer
        
        # 检查书籍结构指代
        book_ref_patterns = [
            (r'本书', '书籍指代:本书'),
            (r'本章', '书籍指代:本章'),
            (r'本节', '书籍指代:本节'),
            (r'上一节', '书籍指代:上一节'),
            (r'下一节', '书籍指代:下一节'),
            (r'第\d+章', '书籍指代:第X章'),
            (r'\d+\.\d+(\.\d+)?节', '书籍指代:X.X节'),
        ]
        
        # 仅检查问句中的书籍结构指代（答案中可能合理提及"本书"等词汇）
        for pattern, reason in book_ref_patterns:
            if re.search(pattern, question):
                return reason
        
        # 检查模糊指代词（问句+答案都检查）
        vague_ref_patterns = [
            (r'文中提到', '模糊指代:文中提到'),
            (r'上文所述', '模糊指代:上文所述'),
            (r'如前所述', '模糊指代:如前所述'),
            (r'根据上述内容', '模糊指代:根据上述内容'),
            (r'上面提到的', '模糊指代:上面提到的'),
        ]
        
        for pattern, reason in vague_ref_patterns:
            if re.search(pattern, combined):
                return reason
        
        # 检查图表引用（问句+答案都检查）
        figure_table_patterns = [
            (r'如图\s*\d', '图表引用:如图X'),
            (r'如表\s*\d', '图表引用:如表X'),
            (r'见图\s*\d', '图表引用:见图X'),
            (r'参见表\s*\d', '图表引用:参见表X'),
            (r'图\s*\d+\.\d+\s*表明', '图表引用:图X表明'),
            (r'表\s*\d+\.\d+\s*中', '图表引用:表X中'),
        ]
        
        for pattern, reason in figure_table_patterns:
            if re.search(pattern, combined):
                return reason
        
        # 检查公式编号引用（问句+答案都检查）
        formula_ref_patterns = [
            (r'公式\s*[\(\uff08]\s*\d', '公式编号:公式(X)'),
            (r'由式\s*[\(\uff08]\s*\d', '公式编号:由式(X)'),
            (r'代入公式', '公式编号:代入公式'),
            (r'根据公式\s*[\(\uff08]', '公式编号:根据公式(X)'),
        ]
        
        for pattern, reason in formula_ref_patterns:
            if re.search(pattern, combined):
                return reason
        
        return None
    
    def _build_glossary_context(self, text: str) -> str:
        """构建领域术语表上下文"""
        if not self.glossary:
            return ""
        
        # 找出文本中出现的术语
        matched_terms = []
        for term, definition in self.glossary.items():
            if term in text:
                matched_terms.append(f"- {term}：{definition}")
        
        if matched_terms:
            return f"【领域术语参考】\n" + "\n".join(matched_terms[:10])
        return ""
    
    def save_samples(
        self, 
        samples: List[SFTSample], 
        output_path: str,
        format: str = "sharegpt"
    ):
        """
        保存SFT样本到文件
        
        Args:
            samples: 样本列表
            output_path: 输出路径
            format: 输出格式 (sharegpt / alpaca / messages)
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
            exist_ok=True
        )
        
        if format == "sharegpt":
            data = self._to_sharegpt(samples)
        elif format == "alpaca":
            data = self._to_alpaca(samples)
        elif format == "messages":
            data = self._to_messages(samples)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        save_jsonl(data, output_path)
        logger.info(f"SFT语料已保存: {output_path} ({len(data)} 条, 格式={format})")
    
    def _to_sharegpt(self, samples: List[SFTSample]) -> List[Dict]:
        """转换为ShareGPT格式"""
        data = []
        for s in samples:
            data.append({
                "type": s.type,
                "source_chunk_id": s.source_chunk_id,
                "conversations": s.conversations,
                "confidence": s.confidence,
                "diagnosis": s.diagnosis,
                "metadata": s.metadata
            })
        return data
    
    def _to_alpaca(self, samples: List[SFTSample]) -> List[Dict]:
        """转换为Alpaca格式"""
        data = []
        for s in samples:
            if len(s.conversations) >= 2:
                data.append({
                    "instruction": s.conversations[0]["value"],
                    "input": "",
                    "output": s.conversations[1]["value"],
                    "type": s.type,
                    "source_chunk_id": s.source_chunk_id,
                    "diagnosis": s.diagnosis
                })
        return data
    
    def _to_messages(self, samples: List[SFTSample]) -> List[Dict]:
        """转换为OpenAI Messages格式"""
        data = []
        for s in samples:
            messages = []
            for conv in s.conversations:
                role = "user" if conv["from"] == "user" else "assistant"
                messages.append({"role": role, "content": conv["value"]})
            data.append({
                "messages": messages,
                "type": s.type,
                "source_chunk_id": s.source_chunk_id
            })
        return data
    
    def get_statistics(self) -> Dict:
        """获取生成统计信息"""
        return {
            **self.stats,
            "model": self.model,
            "domain": self.domain,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold
        }
