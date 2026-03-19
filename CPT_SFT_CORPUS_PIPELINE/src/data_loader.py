"""
Markdown数据加载器 v2.3
=======================
负责读取结构化的Markdown书籍文件，解析其层级结构（章节、小节、段落），
并输出结构化的文档树，供后续分块器使用。

v2.3升级（基于真实数据质量反馈）：
1. 智能章节推断：当一级标题全部为出版结构性内容（书名、出版社、前言等）时，
   自动从二级标题中识别章节编号模式（如"5.1"、"1.1"），将其提升为逻辑章节
2. 出版元信息识别：自动识别封面、出版社、CIP数据等出版结构性一级标题
3. 保留v2.2的所有功能：参考文献分离、文本预处理等
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .utils import (
    setup_logger, clean_text, count_chars, 
    preprocess_md_text, is_reference_text
)

logger = setup_logger(__name__)


@dataclass
class Section:
    """文档节点（章节/小节/段落）"""
    title: str                          # 标题文本
    level: int                          # 标题层级（1=一级标题, 2=二级标题, ...）
    content: str                        # 该节的正文内容（不含子节）
    full_content: str = ""              # 该节的完整内容（含子节）
    children: List['Section'] = field(default_factory=list)
    parent_titles: List[str] = field(default_factory=list)  # 祖先标题链
    start_line: int = 0                 # 在原文中的起始行号
    end_line: int = 0                   # 在原文中的结束行号
    is_reference: bool = False          # 是否为参考文献区域
    metadata: Dict = field(default_factory=dict)


@dataclass
class Document:
    """解析后的文档结构"""
    filename: str                       # 文件名
    book_name: str                      # 书籍名称
    total_chars: int = 0                # 总字符数
    total_sections: int = 0             # 总节数
    root_sections: List[Section] = field(default_factory=list)  # 顶层章节列表
    raw_text: str = ""                  # 原始文本
    metadata: Dict = field(default_factory=dict)


# 参考文献行的特征模式（用于逐行检测）
_REF_LINE_PATTERNS = [
    re.compile(r'\[\s*[JMDC]\s*\]', re.IGNORECASE),           # [J]. [M]. [D]. [C]
    re.compile(r'\d{4}\s*[\.．。,，]\s*\S+.*\[\s*[JMDC]\s*\]', re.IGNORECASE),
    re.compile(r'^\S+[，,]\s*\S+.*\d{4}.*\[\s*[JMDC]\s*\]', re.IGNORECASE),
    re.compile(r'\d{4}\.\s*\S+.*(?:Journal|Proceedings|Conference|Trans\.)', re.IGNORECASE),
    re.compile(r'(?:出版社|Press|Publisher)', re.IGNORECASE),
]

# 出版结构性标题关键词（用于识别非正文的一级标题）
_PUBLISHING_TITLE_KEYWORDS = [
    "前言", "序言", "序", "目录", "致谢", "后记",
    "出版社", "出版", "科学出版", "工业出版", "大学出版",
    "press", "publisher", "publishing",
    "内容简介", "图书在版编目", "cip",
    "作者简介", "作者介绍", "编者的话",
    "附录", "索引", "参考文献",
]

# 章节编号模式（用于从二级标题中识别逻辑章节）
_CHAPTER_NUM_PATTERN = re.compile(
    r'^(\d+)\.\d+'          # "5.1", "1.2" 等
    r'|^第\s*[一二三四五六七八九十\d]+\s*章'  # "第一章", "第5章" 等
    r'|^第\s*[一二三四五六七八九十\d]+\s*节'  # "第一节" 等
    r'|^[一二三四五六七八九十]+[、.]'          # "一、" "二." 等
)

# 从标题中提取章节号的模式
_EXTRACT_CHAPTER_NUM = re.compile(r'^(\d+)\.')


def _is_ref_line(line: str) -> bool:
    """判断单行是否为参考文献条目"""
    line = line.strip()
    if not line or len(line) < 10:
        return False
    for p in _REF_LINE_PATTERNS:
        if p.search(line):
            return True
    return False


def _is_publishing_title(title: str) -> bool:
    """判断标题是否为出版结构性标题（非正文内容）"""
    title_lower = title.lower().strip()
    # 检查关键词匹配
    for kw in _PUBLISHING_TITLE_KEYWORDS:
        if kw in title_lower:
            return True
    # 检查是否为纯书名（通常较短且不含章节编号）
    # 如果标题不包含任何章节编号模式，且字数较少，可能是书名
    return False


def _get_chapter_group(title: str) -> Optional[str]:
    """
    从标题中提取章节分组号。
    例如 "5.1 总体参数预算" -> "5"
         "5.1.1 卫星质量预算" -> "5"
         "1.2 微小卫星特点" -> "1"
    """
    m = _EXTRACT_CHAPTER_NUM.match(title.strip())
    if m:
        return m.group(1)
    return None


class MarkdownLoader:
    """
    Markdown文档加载器 v2.3
    
    功能：
    1. 读取Markdown文件
    2. 执行文本预处理（HTML表格转文本、图片引用清理）
    3. 解析标题层级结构（# ~ ####）
    4. 智能章节推断：识别出版结构性标题，从二级标题推断逻辑章节（v2.3新增）
    5. 自动识别并标记参考文献区域
    6. 分离混合Section中的参考文献部分
    7. 构建文档树（Document -> Section -> Section ...）
    """
    
    # Markdown标题正则
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    # 参考文献区域标题关键词
    REFERENCE_TITLE_KEYWORDS = [
        "参考文献", "references", "bibliography", "参考资料", "引用文献"
    ]
    
    def __init__(
        self, 
        heading_levels: List[int] = None,
        enable_preprocess: bool = True,
        detect_references: bool = True
    ):
        self.heading_levels = heading_levels or [1, 2, 3, 4]
        self.max_level = max(self.heading_levels)
        self.enable_preprocess = enable_preprocess
        self.detect_references = detect_references
    
    def load_file(self, filepath: str) -> Document:
        """加载并解析单个Markdown文件"""
        logger.info(f"正在加载文件: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # 文本预处理管线
        if self.enable_preprocess:
            processed_text = preprocess_md_text(raw_text)
            logger.info(
                f"预处理完成: {len(raw_text)} -> {len(processed_text)} 字符 "
                f"(减少 {len(raw_text) - len(processed_text)} 字符)"
            )
        else:
            processed_text = clean_text(raw_text)
        
        # 提取书籍名称
        filename = os.path.basename(filepath)
        book_name = os.path.splitext(filename)[0]
        
        # 解析文档结构
        sections = self._parse_sections(processed_text)
        
        # v2.3: 智能章节推断
        sections = self._infer_chapter_structure(sections)
        
        # 自动检测并标记参考文献区域
        if self.detect_references:
            self._mark_reference_sections(sections)
            self._split_mixed_reference_sections(sections)
        
        doc = Document(
            filename=filename,
            book_name=book_name,
            total_chars=count_chars(processed_text),
            total_sections=self._count_sections(sections),
            root_sections=sections,
            raw_text=processed_text,
            metadata={
                "source_path": os.path.abspath(filepath),
                "file_size": os.path.getsize(filepath),
                "preprocessed": self.enable_preprocess,
                "original_chars": len(raw_text)
            }
        )
        
        ref_count = self._count_reference_sections(sections)
        if ref_count > 0:
            logger.info(f"检测到 {ref_count} 个参考文献区域")
        
        logger.info(
            f"文件解析完成: {filename}, "
            f"总字符数={doc.total_chars}, "
            f"章节数={doc.total_sections}"
        )
        
        return doc
    
    def load_directory(self, dirpath: str, pattern: str = "*.md") -> List[Document]:
        """加载目录下所有Markdown文件"""
        import glob
        files = sorted(glob.glob(os.path.join(dirpath, pattern)))
        
        if not files:
            logger.warning(f"目录 {dirpath} 下未找到匹配 {pattern} 的文件")
            return []
        
        logger.info(f"找到 {len(files)} 个Markdown文件")
        
        documents = []
        for fp in files:
            try:
                doc = self.load_file(fp)
                documents.append(doc)
            except Exception as e:
                logger.error(f"加载文件失败 {fp}: {e}")
        
        return documents
    
    def _parse_sections(self, text: str) -> List[Section]:
        """解析Markdown文本的层级结构"""
        lines = text.split('\n')
        
        headings = []
        for i, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                if level <= self.max_level:
                    headings.append((i, level, title))
        
        if not headings:
            return [Section(
                title="(无标题)",
                level=0,
                content=text,
                full_content=text,
                start_line=0,
                end_line=len(lines) - 1
            )]
        
        raw_sections = []
        for idx, (line_no, level, title) in enumerate(headings):
            start = line_no + 1
            end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
            
            content_lines = lines[start:end]
            content = '\n'.join(content_lines).strip()
            
            raw_sections.append(Section(
                title=title,
                level=level,
                content=content,
                full_content=content,
                start_line=line_no,
                end_line=end - 1
            ))
        
        # 处理标题之前的前言内容
        if headings[0][0] > 0:
            preface_content = '\n'.join(lines[:headings[0][0]]).strip()
            if preface_content:
                raw_sections.insert(0, Section(
                    title="(前言)",
                    level=0,
                    content=preface_content,
                    full_content=preface_content,
                    start_line=0,
                    end_line=headings[0][0] - 1
                ))
        
        root_sections = self._build_tree(raw_sections)
        self._fill_parent_titles(root_sections, [])
        
        return root_sections
    
    def _infer_chapter_structure(self, sections: List[Section]) -> List[Section]:
        """
        v2.3: 智能章节推断
        
        处理PDF转换MD常见的结构问题：
        - 所有内容被归入一个level=0的"(前言)"根节点
        - 一级标题全部为出版结构性内容（书名、出版社、前言等）
        - 正文的二级标题全部嵌套在出版结构性一级标题（如"# 前言"）下
        
        策略：递归搜索整个文档树，找到包含大量带编号子节点的出版结构性标题，
        将其带编号子节点提取出来，按章号分组创建虚拟章节。
        """
        # 递归收集所有一级标题（可能嵌套在level=0节点下）
        all_h1 = self._collect_sections_at_level(sections, 1)
        
        if not all_h1:
            return self._promote_h2_to_chapters(sections)
        
        # 找到需要重组的出版结构性标题（包含大量带编号子节点的）
        targets = []  # (parent_section_or_none, h1_section)
        for h1 in all_h1:
            if self._is_publishing_or_bookname(h1.title, all_h1):
                numbered_children = [
                    c for c in h1.children 
                    if _CHAPTER_NUM_PATTERN.match(c.title.strip())
                ]
                if len(numbered_children) > 3:
                    targets.append(h1)
                    logger.info(
                        f"[章节推断] 出版结构标题 '{h1.title}' 下有 "
                        f"{len(numbered_children)} 个带编号子节点，需要重组"
                    )
        
        if not targets:
            # 没有需要重组的标题
            return sections
        
        # 执行重组
        return self._restructure_deep(sections, targets)
    
    def _collect_sections_at_level(self, sections: List[Section], level: int) -> List[Section]:
        """递归收集指定层级的所有Section"""
        result = []
        for sec in sections:
            if sec.level == level:
                result.append(sec)
            result.extend(self._collect_sections_at_level(sec.children, level))
        return result
    
    def _is_publishing_or_bookname(self, title: str, all_h1: List[Section]) -> bool:
        """
        判断标题是否为出版结构性标题或书名。
        
        书名的判断策略：
        - 如果一级标题不含章节编号且不是明显的正文标题
        - 如果标题内容与文件名高度相似
        - 如果标题很短（<20字）且不含编号
        """
        if _is_publishing_title(title):
            return True
        
        # 检查是否为书名：不含章节编号、不含技术关键词
        title_stripped = title.strip()
        if _CHAPTER_NUM_PATTERN.match(title_stripped):
            return False  # 有章节编号，是正文标题
        
        # 短标题且无编号，可能是书名
        if len(title_stripped) < 30 and not re.search(r'\d+\.\d+', title_stripped):
            # 如果这个标题下有大量带编号的子节点，说明它不是正文章节
            return True
        
        return False
    
    def _restructure_deep(self, sections: List[Section], targets: List[Section]) -> List[Section]:
        """
        深度重组文档结构：
        从target节点中提取带编号子节点，按章号分组创建虚拟章节，
        然后将虚拟章节提升到顶层。
        """
        target_set = set(id(t) for t in targets)
        orphan_numbered = []
        
        # 从所有target中提取带编号子节点
        for target in targets:
            keep_children = []
            for child in target.children:
                if _CHAPTER_NUM_PATTERN.match(child.title.strip()):
                    orphan_numbered.append(child)
                else:
                    keep_children.append(child)
            target.children = keep_children
            self._compute_full_content(target)
        
        if not orphan_numbered:
            return sections
        
        # 按章号分组
        chapter_groups = self._group_by_chapter(orphan_numbered)
        
        # 创建虚拟章节父节点
        virtual_chapters = []
        for chapter_num, children in sorted(chapter_groups.items(), key=lambda x: x[0]):
            if chapter_num == -1:
                chapter_title = "其他"
            else:
                chapter_title = f"第{chapter_num}章"
            
            virtual_chapter = Section(
                title=chapter_title,
                level=1,
                content="",
                full_content="",
                children=children,
                start_line=children[0].start_line,
                end_line=children[-1].end_line,
                metadata={"virtual": True, "chapter_num": chapter_num}
            )
            
            self._compute_full_content(virtual_chapter)
            virtual_chapters.append(virtual_chapter)
            
            logger.info(
                f"[章节推断] 创建虚拟章节: '{chapter_title}' "
                f"(含 {len(children)} 个子节点)"
            )
        
        # 重建顶层sections列表
        # 策略：保留非target的sections，将虚拟章节插入到正确位置
        new_sections = []
        for sec in sections:
            new_sections.append(sec)
        
        # 将虚拟章节添加到顶层
        new_sections.extend(virtual_chapters)
        
        # 按start_line排序
        new_sections.sort(key=lambda s: s.start_line)
        
        # 重新填充parent_titles
        self._fill_parent_titles(new_sections, [])
        
        return new_sections
    
    def _restructure_sections(self, sections: List[Section]) -> List[Section]:
        """
        重组文档结构（兼容旧接口）：
        1. 出版结构性的一级标题及其非编号子节点 -> 保留为出版元信息
        2. 带章节编号的二级标题 -> 按章号分组，创建虚拟章节父节点
        """
        new_sections = []
        orphan_numbered = []  # 需要重组的带编号子节点
        
        for sec in sections:
            if sec.level == 1 and _is_publishing_title(sec.title):
                # 出版结构性标题：分离其带编号的子节点
                keep_children = []
                for child in sec.children:
                    if _CHAPTER_NUM_PATTERN.match(child.title.strip()):
                        orphan_numbered.append(child)
                    else:
                        keep_children.append(child)
                
                if keep_children or sec.content.strip():
                    sec.children = keep_children
                    self._compute_full_content(sec)
                    new_sections.append(sec)
                # 如果出版结构标题下没有保留任何内容，也保留空壳（后续会被过滤）
                elif not keep_children and not sec.content.strip():
                    new_sections.append(sec)
                else:
                    sec.children = keep_children
                    self._compute_full_content(sec)
                    new_sections.append(sec)
            elif sec.level <= 0:
                # 前言内容（level=0），直接保留
                new_sections.append(sec)
            else:
                # 非出版结构性的一级标题，也检查其子节点
                # 但通常这种情况不会进入_restructure_sections
                new_sections.append(sec)
        
        if not orphan_numbered:
            return new_sections
        
        # 按章号分组
        chapter_groups = self._group_by_chapter(orphan_numbered)
        
        # 创建虚拟章节父节点
        for chapter_num, children in sorted(chapter_groups.items(), key=lambda x: x[0]):
            # 从第一个子节点的标题推断章节名称
            first_title = children[0].title.strip()
            # 提取章节主标题（如 "5.1 总体参数预算" -> 章节5）
            chapter_title = f"第{chapter_num}章"
            
            # 尝试从子节点中找到章节级标题（如 "5.1" 而非 "5.1.1"）
            for child in children:
                t = child.title.strip()
                m = re.match(r'^(\d+)\.(\d+)\s+(.+)$', t)
                if m and m.group(1) == str(chapter_num):
                    # 这是一个 X.Y 级别的标题，可以作为小节
                    pass
            
            virtual_chapter = Section(
                title=chapter_title,
                level=1,
                content="",
                full_content="",
                children=children,
                start_line=children[0].start_line,
                end_line=children[-1].end_line,
                metadata={"virtual": True, "chapter_num": chapter_num}
            )
            
            self._compute_full_content(virtual_chapter)
            new_sections.append(virtual_chapter)
            
            logger.info(
                f"[章节推断] 创建虚拟章节: '{chapter_title}' "
                f"(含 {len(children)} 个子节点)"
            )
        
        # 按start_line排序，保持文档顺序
        new_sections.sort(key=lambda s: s.start_line)
        
        # 重新填充parent_titles
        self._fill_parent_titles(new_sections, [])
        
        return new_sections
    
    def _promote_h2_to_chapters(self, sections: List[Section]) -> List[Section]:
        """当没有一级标题时，从二级标题推断章节结构"""
        # 收集所有二级标题
        all_h2 = []
        for sec in sections:
            if sec.level == 2:
                all_h2.append(sec)
            for child in sec.children:
                if child.level == 2:
                    all_h2.append(child)
        
        if not all_h2:
            return sections
        
        numbered = [s for s in all_h2 if _CHAPTER_NUM_PATTERN.match(s.title.strip())]
        if len(numbered) < 3:
            return sections
        
        chapter_groups = self._group_by_chapter(numbered)
        
        new_sections = []
        for sec in sections:
            if sec.level <= 0:
                new_sections.append(sec)
        
        for chapter_num, children in sorted(chapter_groups.items(), key=lambda x: x[0]):
            virtual_chapter = Section(
                title=f"第{chapter_num}章",
                level=1,
                content="",
                full_content="",
                children=children,
                start_line=children[0].start_line,
                end_line=children[-1].end_line,
                metadata={"virtual": True, "chapter_num": chapter_num}
            )
            self._compute_full_content(virtual_chapter)
            new_sections.append(virtual_chapter)
        
        new_sections.sort(key=lambda s: s.start_line)
        self._fill_parent_titles(new_sections, [])
        
        return new_sections
    
    def _group_by_chapter(self, sections: List[Section]) -> Dict[int, List[Section]]:
        """将带编号的Section按章号分组"""
        groups = {}
        for sec in sections:
            chapter_num_str = _get_chapter_group(sec.title)
            if chapter_num_str:
                chapter_num = int(chapter_num_str)
                if chapter_num not in groups:
                    groups[chapter_num] = []
                groups[chapter_num].append(sec)
            else:
                # 无法提取章号的，归入"其他"组（用-1表示）
                if -1 not in groups:
                    groups[-1] = []
                groups[-1].append(sec)
        return groups
    
    def _build_tree(self, sections: List[Section]) -> List[Section]:
        """将扁平的Section列表构建为层级树"""
        if not sections:
            return []
        
        root = []
        stack = []
        
        for sec in sections:
            while stack and stack[-1][0] >= sec.level:
                stack.pop()
            
            if stack:
                parent = stack[-1][1]
                parent.children.append(sec)
            else:
                root.append(sec)
            
            stack.append((sec.level, sec))
        
        for sec in root:
            self._compute_full_content(sec)
        
        return root
    
    def _compute_full_content(self, section: Section):
        """递归计算节点的完整内容（含子节点）"""
        parts = [section.content] if section.content else []
        
        for child in section.children:
            self._compute_full_content(child)
            child_text = f"{'#' * child.level} {child.title}\n{child.full_content}"
            parts.append(child_text)
        
        section.full_content = '\n\n'.join(parts)
    
    def _fill_parent_titles(self, sections: List[Section], ancestors: List[str]):
        """递归填充每个节点的祖先标题链"""
        for sec in sections:
            sec.parent_titles = list(ancestors)
            self._fill_parent_titles(sec.children, ancestors + [sec.title])
    
    def _mark_reference_sections(self, sections: List[Section]):
        """自动检测并标记参考文献区域"""
        self._mark_references_recursive(sections)
    
    def _mark_references_recursive(self, sections: List[Section]):
        """递归标记参考文献区域"""
        for sec in sections:
            # 策略1：标题匹配
            title_lower = sec.title.lower().strip()
            if any(kw in title_lower for kw in self.REFERENCE_TITLE_KEYWORDS):
                sec.is_reference = True
                self._mark_all_children_as_reference(sec)
                logger.info(f"[参考文献标记] 标题匹配: '{sec.title}'")
                continue
            
            # 策略2：纯参考文献内容检测
            if sec.content and is_reference_text(sec.content):
                ref_boundary = self._find_reference_boundary(sec.content)
                if ref_boundary is not None and ref_boundary > 0:
                    sec.metadata['has_mixed_references'] = True
                    sec.metadata['ref_boundary_line'] = ref_boundary
                    logger.info(
                        f"[参考文献检测] 混合内容: '{sec.title}' "
                        f"(边界在第{ref_boundary}行, 共{len(sec.content.splitlines())}行)"
                    )
                else:
                    sec.is_reference = True
                    logger.info(
                        f"[参考文献标记] 内容检测: '{sec.title}' "
                        f"({count_chars(sec.content)} 字符)"
                    )
            
            self._mark_references_recursive(sec.children)
    
    def _find_reference_boundary(self, content: str) -> Optional[int]:
        """在混合内容中查找参考文献的起始边界"""
        lines = content.split('\n')
        non_empty_lines = [(i, l.strip()) for i, l in enumerate(lines) if l.strip()]
        
        if not non_empty_lines:
            return None
        
        last_non_ref_idx = None
        consecutive_ref_count = 0
        
        for i in range(len(non_empty_lines) - 1, -1, -1):
            line_idx, line_text = non_empty_lines[i]
            if _is_ref_line(line_text):
                consecutive_ref_count += 1
            else:
                last_non_ref_idx = i
                break
        
        if consecutive_ref_count < 3:
            return None
        
        if last_non_ref_idx is None:
            return 0
        
        boundary_non_empty_idx = last_non_ref_idx + 1
        if boundary_non_empty_idx < len(non_empty_lines):
            return non_empty_lines[boundary_non_empty_idx][0]
        
        return None
    
    def _split_mixed_reference_sections(self, sections: List[Section]):
        """分离混合Section中的参考文献部分"""
        self._split_mixed_recursive(sections)
        self._fill_parent_titles(sections, [])
    
    def _split_mixed_recursive(self, sections: List[Section]):
        """递归处理混合Section"""
        for sec in sections:
            if sec.metadata.get('has_mixed_references'):
                self._do_split_section(sec)
            self._split_mixed_recursive(sec.children)
    
    def _do_split_section(self, section: Section):
        """执行Section分离"""
        boundary_line = section.metadata.get('ref_boundary_line')
        if boundary_line is None:
            return
        
        lines = section.content.split('\n')
        
        body_lines = lines[:boundary_line]
        ref_lines = lines[boundary_line:]
        
        body_content = '\n'.join(body_lines).strip()
        ref_content = '\n'.join(ref_lines).strip()
        
        if not ref_content:
            return
        
        section.content = body_content
        section.is_reference = False
        
        ref_section = Section(
            title="参考文献",
            level=section.level + 1,
            content=ref_content,
            full_content=ref_content,
            start_line=section.start_line + boundary_line,
            end_line=section.end_line,
            is_reference=True,
            parent_titles=section.parent_titles + [section.title]
        )
        
        section.children.append(ref_section)
        self._compute_full_content(section)
        
        logger.info(
            f"[参考文献分离] '{section.title}': "
            f"正文={count_chars(body_content)}字, "
            f"参考文献={count_chars(ref_content)}字"
        )
        
        section.metadata.pop('has_mixed_references', None)
        section.metadata.pop('ref_boundary_line', None)
    
    def _mark_all_children_as_reference(self, section: Section):
        """将节点的所有子节点标记为参考文献"""
        for child in section.children:
            child.is_reference = True
            self._mark_all_children_as_reference(child)
    
    def _count_sections(self, sections: List[Section]) -> int:
        """递归统计节数"""
        count = len(sections)
        for sec in sections:
            count += self._count_sections(sec.children)
        return count
    
    def _count_reference_sections(self, sections: List[Section]) -> int:
        """递归统计参考文献区域节数"""
        count = sum(1 for sec in sections if sec.is_reference)
        for sec in sections:
            count += self._count_reference_sections(sec.children)
        return count
    
    def flatten_sections(self, document: Document) -> List[Section]:
        """将文档树展平为Section列表（深度优先遍历）"""
        result = []
        self._flatten_recursive(document.root_sections, result)
        return result
    
    def _flatten_recursive(self, sections: List[Section], result: List[Section]):
        """递归展平"""
        for sec in sections:
            result.append(sec)
            self._flatten_recursive(sec.children, result)
