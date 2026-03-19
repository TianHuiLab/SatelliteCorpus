"""
CPT（Continual Pre-training）语料生成器
========================================
将分块后的书籍内容转换为CPT训练语料。
CPT语料的核心目标是让模型"阅读"领域知识，建立领域语义表征。

CPT语料特点：
- 保留原文的自然语序和表达方式
- 添加结构化的元数据头（章节信息、领域标签）
- 支持多种输出格式（JSONL、纯文本）
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .chunker import Chunk
from .utils import setup_logger, save_jsonl, get_timestamp

logger = setup_logger(__name__)


@dataclass
class CPTSample:
    """CPT训练样本"""
    text: str                           # 训练文本
    source_chunk_id: str                # 来源chunk ID
    chapter: str                        # 所属章节
    section: str                        # 所属小节
    domain: str                         # 领域标签
    char_count: int                     # 字符数
    metadata: Dict = field(default_factory=dict)


class CPTGenerator:
    """
    CPT语料生成器
    
    生成策略：
    1. 基础模式：直接将chunk内容作为CPT语料
    2. 增强模式：添加元数据头，提供结构化上下文
    3. 上下文模式：将相邻chunk合并，提供更完整的语境
    
    输出格式：
    - JSONL：每行一个JSON对象，包含text和metadata字段
    - TXT：纯文本格式，用分隔符分隔不同样本
    """
    
    def __init__(
        self,
        domain: str = "aerospace",
        add_metadata_header: bool = True,
        separator: str = "\n\n",
        output_format: str = "jsonl"
    ):
        self.domain = domain
        self.add_header = add_metadata_header
        self.separator = separator
        self.output_format = output_format
        
        logger.info(f"CPT生成器初始化: domain={domain}, format={output_format}")
    
    def generate(self, chunks: List[Chunk], book_name: str = "") -> List[CPTSample]:
        """
        从chunk列表生成CPT训练样本
        
        Args:
            chunks: 分块列表
            book_name: 书籍名称
            
        Returns:
            List[CPTSample]: CPT样本列表
        """
        logger.info(f"开始生成CPT语料，共 {len(chunks)} 个chunk...")
        
        samples = []
        
        for chunk in chunks:
            # 构建CPT文本
            text = self._build_cpt_text(chunk, book_name)
            
            sample = CPTSample(
                text=text,
                source_chunk_id=chunk.chunk_id,
                chapter=chunk.chapter_title,
                section=chunk.section_title,
                domain=self.domain,
                char_count=len(text),
                metadata={
                    "type_hint": chunk.type_hint,
                    "context_anchor": chunk.context_anchor,
                    "book_name": book_name,
                    "chunk_index": chunk.chunk_index
                }
            )
            samples.append(sample)
        
        logger.info(f"CPT语料生成完成: {len(samples)} 条样本")
        return samples
    
    def generate_contextual(
        self, 
        chunks: List[Chunk], 
        book_name: str = "",
        window_size: int = 3
    ) -> List[CPTSample]:
        """
        生成带上下文窗口的CPT语料
        将相邻的chunk合并，提供更完整的语境
        
        Args:
            chunks: 分块列表
            book_name: 书籍名称
            window_size: 上下文窗口大小（前后各取多少个chunk）
        """
        logger.info(f"生成上下文CPT语料，窗口大小={window_size}...")
        
        samples = []
        
        for i, chunk in enumerate(chunks):
            # 收集窗口内的chunk
            start = max(0, i - window_size)
            end = min(len(chunks), i + window_size + 1)
            window_chunks = chunks[start:end]
            
            # 只合并同章节的chunk
            same_chapter = [c for c in window_chunks if c.chapter_title == chunk.chapter_title]
            
            # 构建合并文本
            parts = []
            if self.add_header:
                parts.append(f"【{self.domain}领域知识】{chunk.context_anchor}")
            
            for c in same_chapter:
                parts.append(c.content)
            
            text = "\n\n".join(parts)
            
            sample = CPTSample(
                text=text,
                source_chunk_id=chunk.chunk_id,
                chapter=chunk.chapter_title,
                section=chunk.section_title,
                domain=self.domain,
                char_count=len(text),
                metadata={
                    "type_hint": chunk.type_hint,
                    "context_window": f"{start}-{end}",
                    "merged_chunks": len(same_chapter),
                    "book_name": book_name
                }
            )
            samples.append(sample)
        
        logger.info(f"上下文CPT语料生成完成: {len(samples)} 条样本")
        return samples
    
    def _build_cpt_text(self, chunk: Chunk, book_name: str) -> str:
        """
        构建单个chunk的CPT文本
        
        格式：
        [元数据头（可选）]
        [正文内容]
        """
        parts = []
        
        if self.add_header:
            # 元数据头：提供结构化上下文
            header_parts = []
            if book_name:
                header_parts.append(f"《{book_name}》")
            header_parts.append(chunk.context_anchor)
            
            header = " | ".join(header_parts)
            parts.append(f"【{self.domain}领域知识】{header}")
        
        # 正文内容
        parts.append(chunk.content)
        
        return "\n\n".join(parts)
    
    def save(self, samples: List[CPTSample], output_path: str):
        """
        保存CPT语料到文件
        
        Args:
            samples: CPT样本列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if self.output_format == "jsonl":
            self._save_jsonl(samples, output_path)
        elif self.output_format == "txt":
            self._save_txt(samples, output_path)
        else:
            raise ValueError(f"不支持的输出格式: {self.output_format}")
        
        logger.info(f"CPT语料已保存: {output_path} ({len(samples)} 条)")
    
    def _save_jsonl(self, samples: List[CPTSample], path: str):
        """保存为JSONL格式"""
        data = []
        for s in samples:
            data.append({
                "text": s.text,
                "source_chunk_id": s.source_chunk_id,
                "chapter": s.chapter,
                "section": s.section,
                "domain": s.domain,
                "char_count": s.char_count,
                "metadata": s.metadata
            })
        save_jsonl(data, path)
    
    def _save_txt(self, samples: List[CPTSample], path: str):
        """保存为纯文本格式"""
        with open(path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(samples):
                f.write(s.text)
                if i < len(samples) - 1:
                    f.write(f"\n{'='*60}\n")
    
    def get_statistics(self, samples: List[CPTSample]) -> Dict:
        """获取CPT语料统计信息"""
        if not samples:
            return {"total": 0}
        
        char_counts = [s.char_count for s in samples]
        chapters = set(s.chapter for s in samples)
        
        return {
            "total_samples": len(samples),
            "total_chars": sum(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "chapters": len(chapters),
            "domain": self.domain
        }
