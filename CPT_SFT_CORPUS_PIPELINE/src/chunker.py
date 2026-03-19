"""
语义分块器 v2.1
===============
基于文档结构和语义边界进行智能分块。
核心原则：分块粒度 = 模型能理解的最小语义单元（通常200-800字），
保留章节标题作为上下文锚点。

v2.1升级：
1. 改进对HTML表格转文本后的分块处理（表格行作为原子单元）
2. 改进对LaTeX公式密集段落的分块（公式块不拆分）
3. 从Section传递is_reference标记到Chunk
4. 对含表格/公式的chunk适当放宽max_chars上限
5. 过滤极短无效chunk（如纯封面信息）
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .data_loader import Section, Document
from .utils import (
    setup_logger, count_chars, detect_content_type, 
    generate_chunk_id, text_hash
)

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """语义分块数据结构"""
    chunk_id: str                       # 唯一标识符
    content: str                        # 分块正文内容
    context_anchor: str                 # 上下文锚点（章节标题链）
    chapter_title: str                  # 所属章节标题
    section_title: str                  # 所属小节标题
    parent_titles: List[str]            # 完整的祖先标题链
    type_hint: str                      # 内容类型初步标签
    char_count: int                     # 字符数
    chunk_index: int                    # 在全书中的序号
    prev_summary: str = ""              # 前一个chunk的摘要（上下文连贯）
    next_summary: str = ""              # 后一个chunk的摘要
    source_line_start: int = 0          # 原文起始行号
    source_line_end: int = 0            # 原文结束行号
    content_hash: str = ""              # 内容哈希（用于去重）
    is_reference: bool = False          # 是否为参考文献区域（v2.1新增）
    has_formula: bool = False           # 是否包含公式（v2.1新增）
    has_table_data: bool = False        # 是否包含表格数据（v2.1新增）
    metadata: Dict = field(default_factory=dict)


class SemanticChunker:
    """
    语义分块器 v2.1
    
    分块策略（优先级从高到低）：
    1. 按章节/小节的自然边界分块
    2. 对超长节进行段落级切分
    3. 对超长段落进行句子级切分
    4. 保证每个chunk在[min_chars, max_chars]范围内
    5. 相邻chunk之间保留overlap_chars字符的重叠
    6. 对含表格/公式的chunk适当放宽max_chars上限（v2.1）
    7. 过滤极短无效chunk（v2.1）
    """
    
    # 段落分隔符
    PARAGRAPH_SEPARATOR = re.compile(r'\n\s*\n')
    # 句子结束标志（中文和英文）
    SENTENCE_END = re.compile(r'([。！？；\.\!\?\;])\s*')
    # LaTeX公式块（行间公式）
    FORMULA_BLOCK = re.compile(r'\$\$.*?\$\$|\\\[.*?\\\]', re.DOTALL)
    # 极短内容阈值（低于此值的chunk可能是封面/版权等无效内容）
    MIN_VALID_CHARS = 15
    
    def __init__(
        self,
        min_chunk_chars: int = 200,
        max_chunk_chars: int = 800,
        overlap_chars: int = 50,
        preserve_context_anchor: bool = True
    ):
        self.min_chars = min_chunk_chars
        self.max_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.preserve_anchor = preserve_context_anchor
        # v2.1: 对含表格/公式的chunk，放宽上限到1.5倍
        self.max_chars_extended = int(max_chunk_chars * 1.5)
        
        logger.info(
            f"分块器初始化: min={self.min_chars}, max={self.max_chars}, "
            f"overlap={self.overlap_chars}, extended_max={self.max_chars_extended}"
        )
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        对整个文档进行分块
        """
        logger.info(f"开始对文档 '{document.book_name}' 进行分块...")
        
        all_chunks = []
        chunk_index = 0
        
        for root_section in document.root_sections:
            section_chunks = self._chunk_section(
                section=root_section,
                book_name=document.book_name,
                start_index=chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # 填充前后文摘要
        self._fill_context_summaries(all_chunks)
        
        # 去重
        all_chunks = self._deduplicate(all_chunks)
        
        # v2.1: 过滤极短无效chunk
        before_filter = len(all_chunks)
        all_chunks = [c for c in all_chunks if c.char_count >= self.MIN_VALID_CHARS]
        if before_filter != len(all_chunks):
            logger.info(
                f"过滤极短chunk: {before_filter} -> {len(all_chunks)} "
                f"(移除 {before_filter - len(all_chunks)} 个)"
            )
        
        # 重新编号
        for i, c in enumerate(all_chunks):
            c.chunk_index = i
        
        # 统计
        type_counts = {}
        for c in all_chunks:
            type_counts[c.type_hint] = type_counts.get(c.type_hint, 0) + 1
        
        ref_count = sum(1 for c in all_chunks if c.is_reference)
        formula_count = sum(1 for c in all_chunks if c.has_formula)
        table_count = sum(1 for c in all_chunks if c.has_table_data)
        
        logger.info(
            f"分块完成: 共 {len(all_chunks)} 个chunk, "
            f"类型分布: {type_counts}, "
            f"参考文献={ref_count}, 含公式={formula_count}, 含表格={table_count}"
        )
        
        return all_chunks
    
    def _chunk_section(
        self, 
        section: Section, 
        book_name: str,
        start_index: int
    ) -> List[Chunk]:
        """
        对单个Section进行分块
        """
        chunks = []
        current_index = start_index
        
        content = section.content.strip()
        if content:
            # v2.1: 检测内容特征，决定是否使用扩展上限
            has_formula = bool(self.FORMULA_BLOCK.search(content))
            has_table_data = self._has_structured_data(content)
            effective_max = self.max_chars_extended if (has_formula or has_table_data) else self.max_chars
            
            content_chunks = self._split_content(content, max_chars=effective_max)
            
            for i, text in enumerate(content_chunks):
                chunk = Chunk(
                    chunk_id=generate_chunk_id(
                        book_name, 
                        section.parent_titles[0] if section.parent_titles else section.title,
                        section.title,
                        current_index
                    ),
                    content=text,
                    context_anchor=self._build_context_anchor(section),
                    chapter_title=section.parent_titles[0] if section.parent_titles else section.title,
                    section_title=section.title,
                    parent_titles=section.parent_titles + [section.title],
                    type_hint=detect_content_type(text),
                    char_count=count_chars(text),
                    chunk_index=current_index,
                    source_line_start=section.start_line,
                    source_line_end=section.end_line,
                    content_hash=text_hash(text),
                    is_reference=getattr(section, 'is_reference', False),
                    has_formula=bool(self.FORMULA_BLOCK.search(text)),
                    has_table_data=self._has_structured_data(text),
                    metadata={
                        "section_level": section.level,
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(content_chunks),
                        "effective_max_chars": effective_max
                    }
                )
                chunks.append(chunk)
                current_index += 1
        
        # 递归处理子Section
        for child in section.children:
            child_chunks = self._chunk_section(child, book_name, current_index)
            chunks.extend(child_chunks)
            current_index += len(child_chunks)
        
        return chunks
    
    def _has_structured_data(self, text: str) -> bool:
        """
        v2.1: 检测文本是否包含结构化数据（表格转换后的文本）
        
        特征：
        - 多行 "字段名: 值" 格式
        - 多行 "；" 分隔的数据
        """
        # 检测 "key: value" 模式
        kv_pattern = re.findall(r'^.+[:：]\s*.+$', text, re.MULTILINE)
        if len(kv_pattern) >= 3:
            return True
        
        # 检测 "；" 分隔的多字段行
        semicolon_lines = [l for l in text.split('\n') if l.count('；') >= 2]
        if len(semicolon_lines) >= 2:
            return True
        
        return False
    
    def _split_content(self, text: str, max_chars: int = None) -> List[str]:
        """
        将内容切分为合适大小的块
        v2.1: 支持动态max_chars参数
        """
        text = text.strip()
        char_count = count_chars(text)
        effective_max = max_chars or self.max_chars
        
        # 文本在合理范围内，直接返回
        if char_count <= effective_max:
            if char_count >= self.min_chars:
                return [text]
            else:
                return [text] if char_count > 0 else []
        
        # 按段落切分
        paragraphs = self.PARAGRAPH_SEPARATOR.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 合并段落为合适大小的chunk
        chunks = self._merge_segments(paragraphs, level="paragraph", max_chars=effective_max)
        
        # 对仍然过长的chunk进行句子级切分
        final_chunks = []
        for chunk_text in chunks:
            if count_chars(chunk_text) > effective_max:
                sentences = self._split_sentences(chunk_text)
                sub_chunks = self._merge_segments(sentences, level="sentence", max_chars=effective_max)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk_text)
        
        return final_chunks
    
    def _merge_segments(
        self, 
        segments: List[str], 
        level: str = "paragraph",
        max_chars: int = None
    ) -> List[str]:
        """
        将小段合并为合适大小的chunk
        v2.1: 支持动态max_chars参数
        """
        if not segments:
            return []
        
        effective_max = max_chars or self.max_chars
        separator = "\n\n" if level == "paragraph" else ""
        chunks = []
        current_parts = []
        current_len = 0
        
        for seg in segments:
            seg_len = count_chars(seg)
            
            if seg_len > effective_max:
                if current_parts:
                    chunks.append(separator.join(current_parts))
                    current_parts = []
                    current_len = 0
                chunks.append(seg)
                continue
            
            new_len = current_len + seg_len + (len(separator) if current_parts else 0)
            
            if new_len > effective_max and current_parts:
                chunks.append(separator.join(current_parts))
                
                if self.overlap_chars > 0 and current_parts:
                    overlap_text = current_parts[-1]
                    if count_chars(overlap_text) <= self.overlap_chars:
                        current_parts = [overlap_text]
                        current_len = count_chars(overlap_text)
                    else:
                        current_parts = []
                        current_len = 0
                else:
                    current_parts = []
                    current_len = 0
            
            current_parts.append(seg)
            current_len = count_chars(separator.join(current_parts))
        
        if current_parts:
            last_chunk = separator.join(current_parts)
            if count_chars(last_chunk) < self.min_chars and chunks:
                prev = chunks[-1]
                merged = prev + separator + last_chunk
                if count_chars(merged) <= effective_max * 1.2:
                    chunks[-1] = merged
                else:
                    chunks.append(last_chunk)
            else:
                chunks.append(last_chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """按句子切分文本"""
        parts = self.SENTENCE_END.split(text)
        
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and self.SENTENCE_END.match(parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _build_context_anchor(self, section: Section) -> str:
        """
        构建上下文锚点字符串
        """
        parts = []
        for i, title in enumerate(section.parent_titles):
            level_name = ["书", "章", "节", "小节", "段"][min(i, 4)]
            parts.append(f"[{level_name}]{title}")
        parts.append(f"[当前]{section.title}")
        return " > ".join(parts)
    
    def _fill_context_summaries(self, chunks: List[Chunk]):
        """填充每个chunk的前后文摘要"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_text = chunks[i - 1].content
                chunk.prev_summary = prev_text[:150] + ("..." if len(prev_text) > 150 else "")
            if i < len(chunks) - 1:
                next_text = chunks[i + 1].content
                chunk.next_summary = next_text[:150] + ("..." if len(next_text) > 150 else "")
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """基于内容哈希去重"""
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.content_hash not in seen:
                seen.add(chunk.content_hash)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"去重: chunk {chunk.chunk_id} 与已有chunk内容重复")
        
        if len(chunks) != len(unique_chunks):
            logger.info(
                f"去重: {len(chunks)} -> {len(unique_chunks)} "
                f"(移除 {len(chunks) - len(unique_chunks)} 个重复)"
            )
        
        return unique_chunks
    
    def get_statistics(self, chunks: List[Chunk]) -> Dict:
        """获取分块统计信息"""
        if not chunks:
            return {"total": 0}
        
        char_counts = [c.char_count for c in chunks]
        type_counts = {}
        for c in chunks:
            type_counts[c.type_hint] = type_counts.get(c.type_hint, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "type_distribution": type_counts,
            "chapters": len(set(c.chapter_title for c in chunks)),
            "reference_chunks": sum(1 for c in chunks if c.is_reference),
            "formula_chunks": sum(1 for c in chunks if c.has_formula),
            "table_chunks": sum(1 for c in chunks if c.has_table_data)
        }
