"""
知识密度过滤器 v2.1
==================
在语料生成之前，对每个chunk进行知识密度评估，
仅允许具有明确专业知识价值的文本进入后续生成流程。

采用"规则引擎预过滤 + LLM精细判断"双重过滤策略：
- 第一层（规则引擎）：零成本快速拦截出版结构文本、过短文本等明确低价值内容
- 第二层（LLM判断）：对规则引擎无法确定的内容，调用LLM进行知识密度精细评估

v2.1升级：
1. 增强参考文献内容模式检测（不依赖标题，直接检测内容中的文献引用格式）
2. 利用Chunk的is_reference标记（由data_loader在解析阶段标记）
3. 增加图片说明纯引用的过滤
4. 优化规则引擎的判断顺序和效率

知识密度分级：
- high   : 包含明确的专业知识（概念定义、技术原理、系统结构、工程流程、技术参数、案例分析等）
- medium : 包含部分知识但密度不高（背景描述中夹杂少量技术信息）
- low    : 无知识价值（出版信息、致谢、目录、纯叙述、参考文献列表等）
"""

import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .chunker import Chunk
from .utils import setup_logger, safe_json_parse, count_chars, is_reference_text

logger = setup_logger(__name__)


# ============================================================
# 规则引擎：低价值内容的关键词与模式库
# ============================================================

# 出版结构章节标题关键词（命中即过滤）
LOW_VALUE_SECTION_KEYWORDS = [
    "前言", "序言", "序", "自序", "代序", "再版前言",
    "目录", "总目录", "详细目录",
    "致谢", "鸣谢", "后记", "跋",
    "作者介绍", "作者简介", "编者简介", "关于作者",
    "出版说明", "出版者的话", "编辑说明", "编者按",
    "版权", "版权页", "图书在版编目",
    "丛书介绍", "丛书序", "丛书总序",
    "索引", "缩略语表",
    "封面", "封底", "扉页",
]

# 出版结构内容的正则模式（命中即过滤）
LOW_VALUE_CONTENT_PATTERNS = [
    r'ISBN\s*[\d\-]+',                              # ISBN号
    r'CIP\s*数据',                                   # CIP编目
    r'图书在版编目',                                  # 编目信息
    r'版权所有.*翻印必究',                            # 版权声明
    r'定价[：:]\s*\d+',                               # 定价信息
    r'印刷.*印次',                                    # 印刷信息
    r'出版社.*出版',                                  # 出版社信息
    r'邮编[：:]\s*\d{6}',                             # 邮编
    r'电话[：:]\s*[\d\-]+',                           # 电话
    r'本书属于.*丛书',                                # 丛书归属
    r'作者.*教授.*博士.*研究',                         # 作者简历
    r'主编.*副主编',                                  # 编委信息
    r'感谢.*支持.*帮助',                              # 致谢套话
    r'在此.*表示.*感谢',                              # 致谢套话
    r'本书.*编写.*过程中',                            # 编写说明
    r'本书适合.*读者',                                # 读者对象
    r'本书可作为.*教材',                              # 教材说明
]

# v2.1: 参考文献内容的正则模式（用于内容级检测）
REFERENCE_CONTENT_PATTERNS = [
    r'\[\s*[JMDC]\s*\][\.\s]',                      # [J]. [M]. [D]. [C].
    r'\[\s*[JMDC]\s*\]/',                            # [C]//
    r'\d{4}\s*[\.\,]\s*\d+\s*[\(\（]\s*\d+\s*[\)\）]', # 2012. 33(7): 期刊格式
    r'(?:Journal|Proceedings|Conference|Trans\.|IEEE|AIAA)', # 英文期刊/会议
    r'宇航学报|航天器工程|航天器环境工程|中国空间科学技术|航空学报', # 中文航天期刊
]

# 高价值知识内容的关键词模式（命中则标记为高知识密度）
HIGH_VALUE_PATTERNS = {
    "概念定义": [
        r'是指', r'定义为', r'被定义为', r'是一种', r'称为',
        r'指的是', r'简称', r'全称', r'英文.*为', r'缩写为',
        r'概念', r'术语', r'是.*的总称',
    ],
    "技术原理": [
        r'原理', r'机理', r'机制', r'工作方式', r'基本原理',
        r'理论', r'模型', r'方程', r'公式', r'定律', r'定理',
        r'算法', r'控制逻辑', r'反馈.*回路',
    ],
    "系统结构": [
        r'系统.*组成', r'主要.*包括', r'由.*组成', r'架构',
        r'模块', r'子系统', r'组件', r'部件', r'结构.*设计',
        r'接口', r'拓扑', r'层次.*结构',
    ],
    "工程流程": [
        r'步骤[一二三四五六七八九十\d]', r'第[一二三四五六七八九十\d]步',
        r'流程', r'工序', r'程序', r'操作.*方法',
        r'实施.*步骤', r'设计.*流程', r'测试.*程序',
    ],
    "技术参数": [
        r'\d+\s*[a-zA-Z]+[/\s]',                    # 带单位的数值
        r'参数', r'指标', r'约束', r'阈值', r'精度',
        r'频率', r'功率', r'效率', r'比冲', r'推力',
        r'温度.*[℃°]', r'压力.*[MPa|Pa|kPa]',
    ],
    "案例分析": [
        r'故障', r'事故', r'案例', r'异常', r'失效',
        r'原因.*分析', r'排查', r'诊断', r'根本原因',
        r'改进.*措施', r'解决.*方案',
    ],
    "技术对比": [
        r'优点', r'缺点', r'优势', r'劣势', r'对比',
        r'比较', r'区别', r'差异', r'相比',
        r'优缺点', r'利弊',
    ],
}

# 最小有效字符数阈值
MIN_VALID_CHARS = 30


@dataclass
class FilterResult:
    """过滤结果"""
    knowledge_density: str          # high / medium / low
    diagnosis: str                  # 知识型 / 描述型 / 低价值
    reason: str                     # 判断原因
    matched_categories: List[str]   # 命中的知识类别
    should_generate: bool           # 是否应该生成语料
    filter_stage: str               # 过滤阶段: rule_engine / llm
    confidence: float = 0.0         # 置信度


class KnowledgeDensityFilter:
    """
    知识密度过滤器 v2.1
    
    双重过滤策略：
    1. 规则引擎预过滤（零API成本）
    2. LLM精细判断（仅对规则引擎无法确定的内容）
    
    v2.1增强：
    - 利用Chunk.is_reference标记快速过滤参考文献
    - 增加内容级参考文献检测（不依赖标题）
    """
    
    def __init__(
        self,
        min_chars: int = MIN_VALID_CHARS,
        high_value_threshold: int = 2,
        enable_llm_filter: bool = True,
        domain: str = "aerospace"
    ):
        self.min_chars = min_chars
        self.high_value_threshold = high_value_threshold
        self.enable_llm = enable_llm_filter
        self.domain = domain
        
        # 编译正则模式
        self._low_value_patterns = [
            re.compile(p) for p in LOW_VALUE_CONTENT_PATTERNS
        ]
        self._high_value_patterns = {}
        for category, patterns in HIGH_VALUE_PATTERNS.items():
            self._high_value_patterns[category] = [
                re.compile(p) for p in patterns
            ]
        # v2.1: 编译参考文献内容模式
        self._reference_patterns = [
            re.compile(p, re.IGNORECASE) for p in REFERENCE_CONTENT_PATTERNS
        ]
        
        # 统计
        self.stats = {
            "total_processed": 0,
            "rule_filtered": 0,
            "rule_passed": 0,
            "llm_filtered": 0,
            "llm_passed": 0,
            "uncertain_count": 0,
            "reference_filtered": 0,
            "density_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        logger.info(
            f"知识密度过滤器v2.1初始化: min_chars={min_chars}, "
            f"high_threshold={high_value_threshold}, llm={enable_llm_filter}"
        )
    
    def filter_chunk(
        self, 
        chunk: Chunk, 
        llm_callback=None
    ) -> FilterResult:
        """
        对单个chunk进行知识密度过滤
        """
        self.stats["total_processed"] += 1
        
        # ========== 第一层：规则引擎预过滤 ==========
        rule_result = self._rule_engine_filter(chunk)
        
        if rule_result is not None:
            self.stats["density_distribution"][rule_result.knowledge_density] += 1
            if rule_result.should_generate:
                self.stats["rule_passed"] += 1
            else:
                self.stats["rule_filtered"] += 1
            return rule_result
        
        # ========== 第二层：LLM精细判断 ==========
        self.stats["uncertain_count"] += 1
        
        if self.enable_llm and llm_callback:
            llm_result = self._llm_filter(chunk, llm_callback)
            self.stats["density_distribution"][llm_result.knowledge_density] += 1
            if llm_result.should_generate:
                self.stats["llm_passed"] += 1
            else:
                self.stats["llm_filtered"] += 1
            return llm_result
        
        # 无LLM可用时，对不确定的内容默认放行
        default_result = FilterResult(
            knowledge_density="medium",
            diagnosis="描述型",
            reason="规则引擎无法确定，LLM不可用，默认放行",
            matched_categories=[],
            should_generate=True,
            filter_stage="rule_engine_default",
            confidence=0.5
        )
        self.stats["density_distribution"]["medium"] += 1
        self.stats["rule_passed"] += 1
        return default_result
    
    def _rule_engine_filter(self, chunk: Chunk) -> Optional[FilterResult]:
        """
        规则引擎预过滤 v2.1
        
        判断顺序优化：
        1. 参考文献标记检查（最快路径）
        2. 过短文本检查
        3. 章节标题关键词检查
        4. 参考文献内容模式检查（v2.1新增）
        5. 低价值内容模式检查
        6. 纯标题chunk检查
        7. 高价值知识内容匹配
        """
        content = chunk.content.strip()
        section_title = chunk.section_title.lower() if chunk.section_title else ""
        chapter_title = chunk.chapter_title.lower() if chunk.chapter_title else ""
        all_titles = " ".join(chunk.parent_titles).lower() if chunk.parent_titles else ""
        
        # ---------- 检查0（v2.1）：Chunk的is_reference标记 ----------
        if getattr(chunk, 'is_reference', False):
            self.stats["reference_filtered"] += 1
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason="由data_loader标记为参考文献区域",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.98
            )
        
        # ---------- 检查1：过短文本 ----------
        char_count = count_chars(content)
        if char_count < self.min_chars:
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason=f"文本过短（{char_count}字 < {self.min_chars}字阈值）",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.95
            )
        
        # ---------- 检查2：章节标题匹配低价值关键词 ----------
        # 注意：只检查section_title（当前小节标题），不检查chapter_title和parent_titles
        # 因为真实PDF转换的MD文件中，正文章节可能被归入"(前言)"等虚拟父节点下
        # 误将chapter_title纳入检查会导致大量正文内容被错误过滤
        for kw in LOW_VALUE_SECTION_KEYWORDS:
            kw_lower = kw.lower()
            # 精确匹配section_title：标题完全等于关键词，或标题以关键词开头
            if section_title == kw_lower or section_title.startswith(kw_lower):
                return FilterResult(
                    knowledge_density="low",
                    diagnosis="低价值",
                    reason=f"小节标题命中低价值关键词: '{kw}'",
                    matched_categories=[],
                    should_generate=False,
                    filter_stage="rule_engine",
                    confidence=0.95
                )
        
        # ---------- 检查3（v2.1）：参考文献内容模式检测 ----------
        ref_hits = sum(
            len(p.findall(content)) for p in self._reference_patterns
        )
        # 如果参考文献特征密度高（每100字符超过1个命中），判定为参考文献
        ref_density = ref_hits / max(char_count / 100, 1)
        if ref_hits >= 3 and ref_density > 0.5:
            self.stats["reference_filtered"] += 1
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason=f"内容匹配参考文献格式（命中{ref_hits}次，密度={ref_density:.1f}/百字）",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.95
            )
        
        # 也使用utils中的is_reference_text做二次确认
        if is_reference_text(content):
            self.stats["reference_filtered"] += 1
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason="内容经行级分析判定为参考文献列表",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.93
            )
        
        # ---------- 检查4：内容匹配低价值模式 ----------
        low_value_hits = 0
        low_value_reasons = []
        for pattern in self._low_value_patterns:
            if pattern.search(content):
                low_value_hits += 1
                low_value_reasons.append(pattern.pattern[:30])
        
        if low_value_hits >= 2:
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason=f"内容命中{low_value_hits}个低价值模式: {', '.join(low_value_reasons[:3])}",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.90
            )
        
        # ---------- 检查5：纯标题chunk（无实质内容） ----------
        content_no_headers = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE).strip()
        if count_chars(content_no_headers) < self.min_chars:
            return FilterResult(
                knowledge_density="low",
                diagnosis="低价值",
                reason="去除标题后无实质内容",
                matched_categories=[],
                should_generate=False,
                filter_stage="rule_engine",
                confidence=0.90
            )
        
        # ---------- 检查6：高价值知识内容匹配 ----------
        matched_categories = []
        total_high_hits = 0
        
        for category, patterns in self._high_value_patterns.items():
            category_hits = sum(1 for p in patterns if p.search(content))
            if category_hits > 0:
                matched_categories.append(category)
                total_high_hits += category_hits
        
        if total_high_hits >= self.high_value_threshold:
            return FilterResult(
                knowledge_density="high",
                diagnosis="知识型",
                reason=f"命中{total_high_hits}个高价值模式，覆盖类别: {', '.join(matched_categories)}",
                matched_categories=matched_categories,
                should_generate=True,
                filter_stage="rule_engine",
                confidence=min(0.6 + total_high_hits * 0.1, 0.95)
            )
        
        # ---------- 规则引擎无法确定 → 交给LLM ----------
        return None
    
    def _llm_filter(self, chunk: Chunk, llm_callback) -> FilterResult:
        """
        LLM精细过滤
        """
        system_prompt = f"""你是一名专业的{self.domain}领域语料工程师。你的任务是判断给定文本的知识密度。

请严格按以下标准判断：

【高知识密度 knowledge_dense】文本包含以下任一类型：
- 概念定义：术语定义、概念界定、参数说明
- 技术原理：工作机理、算法原理、控制逻辑
- 系统结构：系统组成、模块关系、架构设计
- 工程流程：设计流程、实施步骤、操作流程
- 技术参数：指标、约束、工程参数
- 案例分析：工程实例、任务应用、故障分析
- 技术对比：方法比较、优缺点分析

【低知识密度 low_value】文本属于以下类型：
- 出版结构内容：封面、版权页、前言、序言、作者介绍、丛书介绍、出版说明、致谢、目录
- 参考文献列表：文献引用条目（[J]. [M]. [D]. [C]//等格式）
- 低知识文本：单纯描述书籍背景、不包含技术信息的叙述、章节标题、图表标题、过短文本

必须输出纯JSON：
{{"knowledge_density": "high/low", "diagnosis": "知识型/描述型/低价值", "reason": "判断原因（1-2句话）", "confidence": 0.0-1.0}}"""

        user_prompt = f"""请判断以下文本的知识密度：

【章节位置】{chunk.context_anchor}
【章节标题】{chunk.section_title}
【文本内容】
{chunk.content}"""
        
        response = llm_callback(system_prompt, user_prompt)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed and "knowledge_density" in parsed:
                density = parsed["knowledge_density"]
                is_high = density in ("high", "knowledge_dense")
                
                return FilterResult(
                    knowledge_density="high" if is_high else "low",
                    diagnosis=parsed.get("diagnosis", "描述型"),
                    reason=parsed.get("reason", "LLM判断"),
                    matched_categories=[],
                    should_generate=is_high,
                    filter_stage="llm",
                    confidence=parsed.get("confidence", 0.7)
                )
        
        # LLM失败时默认放行
        logger.warning(f"LLM过滤失败，默认放行: {chunk.chunk_id}")
        return FilterResult(
            knowledge_density="medium",
            diagnosis="描述型",
            reason="LLM判断失败，默认放行",
            matched_categories=[],
            should_generate=True,
            filter_stage="llm_fallback",
            confidence=0.5
        )
    
    def filter_chunks(
        self, 
        chunks: List[Chunk], 
        llm_callback=None,
        progress_callback=None
    ) -> Tuple[List[Chunk], List[Chunk], List[Dict]]:
        """
        批量过滤chunk列表
        """
        logger.info(f"开始知识密度过滤v2.1，共 {len(chunks)} 个chunk...")
        
        passed_chunks = []
        filtered_chunks = []
        filter_details = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks), chunk.chunk_id)
            
            result = self.filter_chunk(chunk, llm_callback)
            
            detail = {
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "char_count": chunk.char_count,
                "knowledge_density": result.knowledge_density,
                "diagnosis": result.diagnosis,
                "reason": result.reason,
                "should_generate": result.should_generate,
                "filter_stage": result.filter_stage,
                "confidence": result.confidence,
                "matched_categories": result.matched_categories,
                "is_reference": getattr(chunk, 'is_reference', False)
            }
            filter_details.append(detail)
            
            if result.should_generate:
                passed_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)
                logger.info(
                    f"[过滤] {chunk.chunk_id} | "
                    f"密度={result.knowledge_density} | "
                    f"原因={result.reason}"
                )
        
        logger.info(
            f"知识密度过滤完成: "
            f"通过={len(passed_chunks)}, "
            f"过滤={len(filtered_chunks)}, "
            f"过滤率={len(filtered_chunks)/len(chunks)*100:.1f}%"
            if chunks else "无chunk"
        )
        
        return passed_chunks, filtered_chunks, filter_details
    
    def get_statistics(self) -> Dict:
        """获取过滤统计信息"""
        total = self.stats["total_processed"]
        return {
            **self.stats,
            "filter_rate": (
                (self.stats["rule_filtered"] + self.stats["llm_filtered"]) / total * 100
                if total > 0 else 0
            ),
            "rule_filter_rate": (
                self.stats["rule_filtered"] / total * 100
                if total > 0 else 0
            ),
            "reference_filter_count": self.stats["reference_filtered"],
        }
