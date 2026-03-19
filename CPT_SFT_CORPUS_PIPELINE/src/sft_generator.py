"""
SFT（Supervised Fine-Tuning）语料动态生成器 v3.1
=================================================
v3.1 优化要点：
1. 【合并调用】将"内容诊断"与"定向生成"从2次API调用合并为1次，
   减少约50%的API调用次数和Token消耗。
2. 【多线程并发】通过 max_workers 参数支持多线程并行处理，
   显著缩短大批量chunk的总处理时间。
3. 【线程安全统计】使用 threading.Lock 保护共享统计字典，
   确保多线程下数据不竞争。

支持的语料类型（完整6种）：
1. fact_qa            - 基础事实问答（适合定义型/描述型内容）
2. comprehension      - 理解型问题（适合需要理解技术原理的内容）
3. reasoning_qa       - 推理问题（适合多步推理/案例分析内容）
4. summarization      - 总结任务（适合信息密集的描述型内容）
5. information_extraction - 信息抽取任务（适合参数/结构密集内容）
6. calculation        - 计算问题（适合含公式推导的内容）

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
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# v3.1 核心优化：诊断+生成一体化Prompt（单次API调用完成两步任务）
SYSTEM_PROMPT_MERGED = """你是一名专业的{domain}领域教材语料工程师。请对给定文本完成"内容诊断 + 语料生成"一体化任务。

【第一步：内容诊断】
分析文本核心属性（单选），并自动对应推荐语料类型：
- 定义型（含术语定义、概念界定、参数说明） → 推荐 fact_qa
- 案例型（含故障分析、事故案例、原因-结论） → 推荐 reasoning_qa
- 过程型（含操作步骤、工艺流程、设计方法） → 推荐 comprehension
- 描述型（事实陈述、背景介绍、系统描述）   → 推荐 summarization
- 数据型（含数据表格、参数对比、统计指标） → 推荐 information_extraction
- 若文本含可推导的数学公式               → 可追加 calculation

判断标准：
- 包含"是指"、"定义为"、"称为"、"分为"等定义性表述 → 定义型
- 包含故障、异常、原因分析、解决方案等 → 案例型
- 包含步骤、流程、程序、操作方法等 → 过程型
- 包含具体数值、参数表、指标对比等 → 数据型
- 主要是背景介绍、历史叙述、一般性描述时 → 描述型

【第二步：定向生成】
根据诊断结果，选择1-2种最合适的语料类型，生成{n}组高质量微调样本。

各类型生成要求：
- fact_qa: 事实性问答，问题形式为"是什么/有哪些/...的定义是什么"
- comprehension: 理解型问答，问题形式为"为什么/作用是什么/如何工作的"
- reasoning_qa: 推理型问答，答案必须用"步骤1：...\\n步骤2：...\\n结论："组织
- summarization: 先从原文提取适合总结的语料片段，再给出总结结果
- information_extraction: 结构化列出关键技术参数/系统组成/性能指标
- calculation: 含完整计算过程，所有公式必须用LaTeX表达

【硬性约束 - 必须严格遵守】
1. 所有内容必须严格源自原文，禁止编造任何事实、数据或结论
2. 问句和答案必须完全自包含，脱离原文后仍可独立理解
3. 严禁出现以下内容：
   - 书籍结构指代："本书"、"第X章"、"X.X节"、"本章"、"本节"、"上一节"、"下一节"
   - 模糊指代词："文中提到"、"上文所述"、"如前所述"、"根据上述内容"
   - 图表引用："如图X所示"、"如表X所示"、"见图X"、"参见表X"
   - 公式编号引用："公式(X.X)"、"由式(X)可得"、"代入公式X"
4. 如涉及公式，必须输出完整LaTeX数学表达式，而非引用编号
5. 生成样本总token数控制在500左右

{negative_example}
{glossary_context}

【输出格式】
必须输出纯JSON，格式如下：
{{"diagnosis": "定义型/案例型/过程型/描述型/数据型", "confidence": 0.0-1.0, "reason": "判断理由（1句话）", "selected_types": ["type1", "type2"], "samples": [{{"type": "fact_qa", "question": "问题", "answer": "回答"}}, {{"type": "summarization", "question": "请总结下面这段描述：...", "input": "待总结语料", "answer": "总结结果"}}]}}"""


# 负面示例（注入到主Prompt）
NEGATIVE_EXAMPLE = """【负面示例 - 请避免】
错误做法：编造原文未提及的解决方案或数据
正确做法：仅基于原文内容进行分析和回答

错误示例：
问：某型号发动机推力不足的原因？
答：可能是燃料纯度不够（❌ 原文未提及燃料纯度问题）

正确示例：
问：某型号发动机推力不足的原因？
答：根据原文分析，推力不足的主要原因是涡轮泵转速下降，导致燃料供给流量不足。（✅ 严格基于原文）"""


# 类型名称标准化映射（修复LLM返回非标准类型名的问题）
TYPE_NORMALIZE = {
    # 中文类型名 → 标准英文名
    "基础事实问答": "fact_qa",
    "事实问答": "fact_qa",
    "理解型问题": "comprehension",
    "推理问题": "reasoning_qa",
    "总结任务": "summarization",
    "总结": "summarization",
    "信息抽取任务": "information_extraction",
    "信息抽取": "information_extraction",
    "计算问题": "calculation",
    # 非标准英文名 → 标准英文名
    "factqa": "fact_qa",
    "fact": "fact_qa",
    "comprehend": "comprehension",
    "reasoning": "reasoning_qa",
    "summary": "summarization",
    "info_extraction": "information_extraction",
    "extract": "information_extraction",
    "calc": "calculation",
    # 兼容旧版类型名（v2.x → v3.x）
    "knowledge_qa": "fact_qa",
    "concept_explanation": "comprehension",
    "text_summary": "summarization",
    "causal_reasoning": "reasoning_qa",
    "cot_reasoning": "reasoning_qa",
    "process_qa": "comprehension",
}

# 标准语料类型集合
VALID_TYPES = {"fact_qa", "comprehension", "reasoning_qa",
               "summarization", "information_extraction", "calculation"}

# 内容诊断类型 → 默认语料类型映射（作为fallback）
TYPE_MAPPING = {
    # 中文诊断标签
    "定义型": ["fact_qa"],
    "案例型": ["reasoning_qa"],
    "过程型": ["comprehension"],
    "描述型": ["summarization"],
    "数据型": ["information_extraction"],
    # 英文兼容
    "definition": ["fact_qa"],
    "case": ["reasoning_qa"],
    "process": ["comprehension"],
    "description": ["summarization"],
    "data": ["information_extraction"],
    # 知识密度过滤器的知识子类别映射
    "概念定义": ["fact_qa"],
    "技术原理": ["reasoning_qa"],
    "系统结构": ["comprehension"],
    "工程流程": ["comprehension"],
    "技术参数": ["information_extraction"],
    "案例分析": ["reasoning_qa"],
    "技术对比": ["summarization"],
    "知识型": ["fact_qa"],
}


def normalize_type(type_name: str) -> Optional[str]:
    """将LLM返回的类型名称标准化为合法的语料类型"""
    if type_name in VALID_TYPES:
        return type_name
    normalized = TYPE_NORMALIZE.get(type_name)
    if normalized:
        return normalized
    # 模糊匹配：检查是否包含关键词
    type_lower = type_name.lower().replace("-", "_").replace(" ", "_")
    for key, val in TYPE_NORMALIZE.items():
        if key.lower() in type_lower or type_lower in key.lower():
            return val
    return None


@dataclass
class SFTSample:
    """SFT训练样本"""
    type: str                           # 语料类型（6种之一）
    source_chunk_id: str                # 来源chunk ID
    conversations: List[Dict]           # 对话列表 [{"from": "user/assistant", "value": "..."}]
    confidence: float = 0.0             # 置信度
    diagnosis: str = ""                 # 内容诊断结果
    metadata: Dict = field(default_factory=dict)


class SFTGenerator:
    """
    SFT语料动态生成器 v3.1

    v3.1核心优化：
    1. 合并调用：diagnose_and_generate_merged() 用单次API调用
       同时完成内容诊断和语料生成，减少~50%的API调用次数
    2. 多线程并发：process_chunks() 支持 max_workers 参数，
       通过 ThreadPoolExecutor 并行处理多个chunk
    3. 线程安全：所有 self.stats 的更新通过 threading.Lock 保护

    支持的LLM API：
    - DashScope（Qwen3-Max）：通过OpenAI兼容接口调用
    - OpenAI兼容接口：支持任何兼容OpenAI API的服务
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-max",
        provider: str = "dashscope",        # v3.1+: API提供商标识（影响 enable_thinking 等特性）
        temperature: float = 0.3,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        batch_delay: float = 0.5,
        domain: str = "aerospace",
        confidence_threshold: float = 0.7,
        max_samples_per_chunk: int = 3,
        enable_negative_examples: bool = True,
        enable_thinking: bool = False,
        enable_llm_filter: bool = True,
        glossary: Dict[str, str] = None,
        max_workers: int = 1,           # v3.1新增：并发线程数
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
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
        self.max_workers = max_workers

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

        # v3.1：线程安全统计锁
        self._stats_lock = threading.Lock()

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
            "self_contained_violations": 0,
            "type_counts": {},
            "filter_stats": {}
        }

        logger.info(
            f"SFT生成器v3.1初始化: provider={provider}, model={model}, domain={domain}, "
            f"temperature={temperature}, max_workers={max_workers}, "
            f"llm_filter={enable_llm_filter}"
        )

    # ============================================================
    # 内部工具方法
    # ============================================================

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

    def _incr_stat(self, key: str, amount: int = 1):
        """线程安全地增加统计计数"""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def _incr_type_count(self, stype: str):
        """线程安全地增加语料类型计数"""
        with self._stats_lock:
            self.stats["type_counts"][stype] = \
                self.stats["type_counts"].get(stype, 0) + 1

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        调用LLM API（带重试机制）

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

                # Qwen3思考模式（仅 dashscope provider 支持）
                if self.enable_thinking and self.provider == "dashscope":
                    kwargs["extra_body"] = {"enable_thinking": True}

                response = self.client.chat.completions.create(**kwargs)

                self._incr_stat("total_api_calls")
                if response.usage:
                    self._incr_stat("total_tokens_used", response.usage.total_tokens)

                content = response.choices[0].message.content
                return content

            except Exception as e:
                logger.warning(
                    f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        self._incr_stat("failed_generations")
        return None

    def _build_glossary_context(self, text: str) -> str:
        """构建领域术语表上下文"""
        if not self.glossary:
            return ""
        matched_terms = []
        for term, definition in self.glossary.items():
            if term in text:
                matched_terms.append(f"- {term}：{definition}")
        if matched_terms:
            return "【领域术语参考】\n" + "\n".join(matched_terms[:10])
        return ""

    def _check_self_contained(self, question: str, answer: str) -> Optional[str]:
        """
        检查问句和答案是否违反自包含约束。

        Returns:
            违反原因字符串，如果没有违反则返回None
        """
        combined = question + " " + answer

        # 检查书籍结构指代（仅检查问句）
        book_ref_patterns = [
            (r'本书', '书籍指代:本书'),
            (r'本章', '书籍指代:本章'),
            (r'本节', '书籍指代:本节'),
            (r'上一节', '书籍指代:上一节'),
            (r'下一节', '书籍指代:下一节'),
            (r'第\d+章', '书籍指代:第X章'),
            (r'\d+\.\d+(\.\d+)?节', '书籍指代:X.X节'),
        ]
        for pattern, reason in book_ref_patterns:
            if re.search(pattern, question):
                return reason

        # 检查模糊指代词（问句+答案）
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

        # 检查图表引用
        figure_table_patterns = [
            (r'如图\s*\d', '图表引用:如图X'),
            (r'如表\s*\d', '图表引用:如表X'),
            (r'见图\s*\d', '图表引用:见图X'),
            (r'参见表\s*\d', '图表引用:参见表X'),
            (r'图\s*\d+\.\d+\s*表明', '图表引用:图X表明'),
        ]
        for pattern, reason in figure_table_patterns:
            if re.search(pattern, combined):
                return reason

        # 检查公式编号引用
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

    # ============================================================
    # v3.1 核心优化：一体化诊断+生成（单次API调用）
    # ============================================================

    def diagnose_and_generate_merged(
        self,
        chunk: Chunk,
        filter_result: FilterResult = None
    ) -> Tuple[Dict, List[SFTSample]]:
        """
        v3.1核心方法：将诊断与生成合并为单次API调用。

        相比v2.x的两步调用（diagnose→generate×N），本方法在一个Prompt中
        完成：内容诊断 + 类型选择 + 样本生成，减少约50%的API调用次数。

        Args:
            chunk: 待处理的chunk
            filter_result: 知识密度过滤结果（可选，用于提供辅助信息）

        Returns:
            Tuple[Dict, List[SFTSample]]:
                - diagnosis_info: 诊断信息字典
                - samples: 生成的SFT样本列表
        """
        negative_example = NEGATIVE_EXAMPLE if self.enable_negative else ""
        glossary_context = self._build_glossary_context(chunk.content)

        system_prompt = SYSTEM_PROMPT_MERGED.format(
            domain=self.domain,
            n=self.max_samples,
            negative_example=negative_example,
            glossary_context=glossary_context
        )

        # 构建用户Prompt，附加过滤器预分析结果作为辅助参考
        filter_hint = ""
        if filter_result and filter_result.matched_categories:
            filter_hint = (
                f"\n\n【预分析参考】规则引擎识别该文本可能涉及以下知识类别："
                f"{', '.join(filter_result.matched_categories)}，请综合判断。"
            )

        user_prompt = (
            f"【章节信息】\n"
            f"章节：{chunk.chapter_title}\n"
            f"小节：{chunk.section_title}\n"
            f"上下文锚点：{chunk.context_anchor}\n\n"
            f"【文本内容】\n{chunk.content}"
            f"{filter_hint}"
        )

        response = self._call_llm(system_prompt, user_prompt)

        # ---------- 解析响应 ----------
        diagnosis_info = {
            "chunk_id": chunk.chunk_id,
            "content_diagnosis": "描述型",
            "content_confidence": 0.5,
            "reason": "",
            "selected_types": [],
        }

        if not response:
            logger.warning(f"合并调用API失败: {chunk.chunk_id}，使用规则引擎fallback")
            diagnosis_info["reason"] = "API不可用，规则引擎fallback"
            fallback_type = TYPE_MAPPING.get(chunk.type_hint or "描述型", ["fact_qa"])
            diagnosis_info["selected_types"] = fallback_type
            return diagnosis_info, []

        parsed = safe_json_parse(response)

        if not parsed:
            logger.warning(
                f"合并调用响应解析失败: {chunk.chunk_id}, "
                f"response={response[:200]}"
            )
            self._incr_stat("failed_generations")
            return diagnosis_info, []

        # 提取诊断信息
        diagnosis_info["content_diagnosis"] = parsed.get("diagnosis", "描述型")
        diagnosis_info["content_confidence"] = parsed.get("confidence", 0.5)
        diagnosis_info["reason"] = parsed.get("reason", "")

        # 标准化 selected_types
        raw_types = parsed.get("selected_types", [])
        normalized_types = []
        for t in raw_types:
            nt = normalize_type(t)
            if nt:
                normalized_types.append(nt)
            else:
                logger.warning(f"无法识别的语料类型 '{t}'，已跳过")
        # 去重并保序
        normalized_types = list(dict.fromkeys(normalized_types))

        # 确保至少有一种类型（fallback）
        if not normalized_types:
            diagnosis = diagnosis_info["content_diagnosis"]
            normalized_types = TYPE_MAPPING.get(diagnosis, ["fact_qa"])

        diagnosis_info["selected_types"] = normalized_types

        # ---------- 解析生成的样本 ----------
        raw_samples = parsed.get("samples", [])
        samples = []

        for sample_data in raw_samples:
            stype = normalize_type(sample_data.get("type", ""))
            if not stype:
                continue

            question = sample_data.get("question", "")
            answer = sample_data.get("answer", "")

            # 防御性处理：LLM有时将字段返回为list而非str
            if isinstance(question, list):
                question = "\n".join(str(q) for q in question)
            elif not isinstance(question, str):
                question = str(question)

            if isinstance(answer, list):
                answer = "\n".join(str(a) for a in answer)
            elif not isinstance(answer, str):
                answer = str(answer)

            if not question.strip() or not answer.strip():
                continue

            # 质量检查：答案过短
            if len(answer.strip()) < 10:
                logger.debug(f"答案过短，跳过: {answer[:50]}")
                continue

            # 质量检查：总长度控制
            if len(question) + len(answer) > 1000:
                logger.debug(f"QA对过长，跳过: Q={question[:30]}...")
                continue

            # 质量检查：自包含约束
            violation = self._check_self_contained(question, answer)
            if violation:
                logger.warning(
                    f"自包含约束违反，跳过: [{violation}] Q={question[:60]}..."
                )
                self._incr_stat("self_contained_violations")
                continue

            # 构建对话（ShareGPT格式基础）
            conversations = [
                {"from": "user", "value": question},
                {"from": "assistant", "value": answer}
            ]

            # 构建精简元数据
            metadata = {
                "chapter": chunk.chapter_title,
                "section": chunk.section_title,
                "context_anchor": chunk.context_anchor,
                "diagnosis_reason": diagnosis_info.get("reason", ""),
            }
            if filter_result:
                metadata["knowledge_density"] = filter_result.knowledge_density
                metadata["matched_categories"] = filter_result.matched_categories

            sample = SFTSample(
                type=stype,
                source_chunk_id=chunk.chunk_id,
                conversations=conversations,
                confidence=diagnosis_info["content_confidence"],
                diagnosis=diagnosis_info["content_diagnosis"],
                metadata=metadata
            )
            samples.append(sample)

            self._incr_stat("successful_generations")
            self._incr_type_count(stype)

        logger.info(
            f"[合并生成] {chunk.chunk_id} | "
            f"诊断={diagnosis_info['content_diagnosis']} | "
            f"类型={normalized_types} | "
            f"样本数={len(samples)}"
        )

        return diagnosis_info, samples

    # ============================================================
    # chunk处理主流程
    # ============================================================

    def process_chunk(self, chunk: Chunk) -> Tuple[List[SFTSample], Dict]:
        """
        处理单个chunk的完整流程：过滤 → 合并诊断+生成

        Returns:
            Tuple[List[SFTSample], Dict]: (生成的样本列表, 诊断+过滤信息)
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
            self._incr_stat("chunks_filtered")
            return [], diagnosis_info

        self._incr_stat("chunks_passed")
        logger.info(
            f"[通过] {chunk.chunk_id} | "
            f"密度={filter_result.knowledge_density} | "
            f"类别={filter_result.matched_categories}"
        )

        # ========== v3.1核心：合并诊断+生成（单次API调用） ==========
        merged_diagnosis, samples = self.diagnose_and_generate_merged(
            chunk, filter_result
        )

        # 更新诊断信息
        diagnosis_info.update({
            "content_diagnosis": merged_diagnosis.get("content_diagnosis", ""),
            "content_confidence": merged_diagnosis.get("content_confidence", 0),
            "content_reason": merged_diagnosis.get("reason", ""),
            "selected_types": merged_diagnosis.get("selected_types", [])
        })

        # 低置信度计数
        if merged_diagnosis.get("content_confidence", 0) < self.confidence_threshold:
            self._incr_stat("low_confidence_count")
            logger.warning(
                f"低置信度chunk: {chunk.chunk_id} "
                f"(confidence={merged_diagnosis.get('content_confidence', 0):.2f})"
            )

        logger.info(f"[完成] {chunk.chunk_id} | 生成 {len(samples)} 条样本")
        return samples, diagnosis_info

    # ============================================================
    # 批量处理（顺序 + 并行）
    # ============================================================

    def process_chunks(
        self,
        chunks: List[Chunk],
        progress_callback=None,
        max_workers: int = None
    ) -> Tuple[List[SFTSample], List[SFTSample], List[Dict]]:
        """
        批量处理chunk列表。

        v3.1：根据 max_workers 自动选择顺序或并行模式。

        Args:
            chunks: chunk列表
            progress_callback: 进度回调 callback(current, total, chunk_id)
            max_workers: 并发线程数（None则使用初始化时的配置）

        Returns:
            Tuple: (合格样本列表, 低置信度样本列表, 诊断结果列表)
        """
        workers = max_workers if max_workers is not None else self.max_workers
        self.stats["total_chunks_input"] = len(chunks)
        logger.info(
            f"开始批量处理 {len(chunks)} 个chunk | "
            f"模式={'并行' if workers > 1 else '顺序'} | "
            f"workers={workers}"
        )

        if workers > 1:
            return self._process_parallel(chunks, progress_callback, workers)
        else:
            return self._process_sequential(chunks, progress_callback)

    def _process_sequential(
        self,
        chunks: List[Chunk],
        progress_callback=None
    ) -> Tuple[List[SFTSample], List[SFTSample], List[Dict]]:
        """顺序处理（单线程，兼容v2.x行为）"""
        all_samples = []
        low_confidence_samples = []
        all_diagnoses = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks), chunk.chunk_id)

            try:
                samples, diagnosis = self.process_chunk(chunk)
                all_diagnoses.append(diagnosis)

                for s in samples:
                    if s.confidence >= self.confidence_threshold:
                        all_samples.append(s)
                    else:
                        low_confidence_samples.append(s)

            except Exception as e:
                logger.error(f"处理chunk失败 {chunk.chunk_id}: {e}")
                self._incr_stat("failed_generations")

            # 批次间延迟（避免API限流）
            if i < len(chunks) - 1 and self.batch_delay > 0:
                time.sleep(self.batch_delay)

        self.stats["filter_stats"] = self.knowledge_filter.get_statistics()
        self._log_batch_summary(all_samples, low_confidence_samples)
        return all_samples, low_confidence_samples, all_diagnoses

    def _process_parallel(
        self,
        chunks: List[Chunk],
        progress_callback=None,
        max_workers: int = 4
    ) -> Tuple[List[SFTSample], List[SFTSample], List[Dict]]:
        """
        并行处理（多线程）。

        v3.1新增：使用 ThreadPoolExecutor 并发调用 process_chunk()。
        注意：self.stats 的所有更新均通过 self._stats_lock 保护，线程安全。

        建议 max_workers 值：
        - DashScope Qwen3：3-5（取决于账号QPM限额）
        - OpenAI：3-8
        """
        all_samples = []
        low_confidence_samples = []
        all_diagnoses = []
        completed_count = [0]   # 使用列表以便在内层函数修改
        results_lock = threading.Lock()

        def _worker(chunk: Chunk):
            """每个线程执行的工作函数"""
            try:
                samples, diagnosis = self.process_chunk(chunk)
                return chunk.chunk_id, samples, diagnosis, None
            except Exception as e:
                logger.error(f"[并行] 处理chunk失败 {chunk.chunk_id}: {e}")
                return chunk.chunk_id, [], {"chunk_id": chunk.chunk_id}, e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(_worker, chunk): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk_id, samples, diagnosis, error = future.result()

                with results_lock:
                    completed_count[0] += 1
                    current = completed_count[0]

                    if progress_callback:
                        progress_callback(current, len(chunks), chunk_id)

                    all_diagnoses.append(diagnosis)

                    if error:
                        self._incr_stat("failed_generations")
                    else:
                        for s in samples:
                            if s.confidence >= self.confidence_threshold:
                                all_samples.append(s)
                            else:
                                low_confidence_samples.append(s)

        self.stats["filter_stats"] = self.knowledge_filter.get_statistics()
        self._log_batch_summary(all_samples, low_confidence_samples)
        return all_samples, low_confidence_samples, all_diagnoses

    def _log_batch_summary(
        self,
        good_samples: List[SFTSample],
        low_conf_samples: List[SFTSample]
    ):
        """输出批处理统计摘要"""
        logger.info(
            f"批量处理完成: "
            f"输入={self.stats['total_chunks_input']}, "
            f"过滤={self.stats['chunks_filtered']}, "
            f"通过={self.stats['chunks_passed']}, "
            f"合格样本={len(good_samples)}, "
            f"低置信度={len(low_conf_samples)}, "
            f"总API调用={self.stats['total_api_calls']}"
        )

    # ============================================================
    # 样本保存
    # ============================================================

    def save_samples(
        self,
        samples: List[SFTSample],
        output_path: str,
        format: str = "sharegpt"
    ):
        """保存SFT样本到文件"""
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
        with self._stats_lock:
            return {
                **self.stats,
                "provider": self.provider,
                "model": self.model,
                "domain": self.domain,
                "temperature": self.temperature,
                "confidence_threshold": self.confidence_threshold,
                "max_workers": self.max_workers
            }
