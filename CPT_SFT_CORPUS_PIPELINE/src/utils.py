"""
工具函数模块 v2.1
================
提供日志、文件IO、文本处理等通用工具函数。
v2.1新增：HTML表格转文本、图片引用清理、参考文献检测等预处理函数。
"""

import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional


def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """配置并返回Logger实例"""
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def count_chars(text: str) -> int:
    """统计文本有效字符数（去除空白）"""
    return len(text.strip())


def count_chinese_chars(text: str) -> int:
    """统计中文字符数"""
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def text_hash(text: str) -> str:
    """计算文本的MD5哈希值（用于去重）"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()


# ============================================================
# 文本清洗与预处理（v2.1增强）
# ============================================================

def clean_text(text: str) -> str:
    """
    清洗文本：去除多余空行、标准化空白
    v2.1: 增加对常见OCR/PDF转换残留的清理
    """
    # 去除连续空行（保留最多一个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除行尾空白
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # 去除零宽字符
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


def html_table_to_text(html_content: str) -> str:
    """
    将HTML表格转换为可读的纯文本格式。
    
    策略：
    1. 提取表头和数据行
    2. 转换为 "字段名: 值" 的结构化文本
    3. 保留表格的语义信息
    
    Args:
        html_content: 包含HTML表格的文本
        
    Returns:
        转换后的文本（表格部分被替换为纯文本）
    """
    result = html_content
    
    # 匹配所有<table>...</table>
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    
    for table_match in table_pattern.finditer(html_content):
        table_html = table_match.group(0)
        table_inner = table_match.group(1)
        
        # 提取所有行
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_inner, re.DOTALL | re.IGNORECASE)
        
        if not rows:
            continue
        
        parsed_rows = []
        for row_html in rows:
            # 提取单元格（th或td）
            cells = re.findall(
                r'<(?:th|td)[^>]*>(.*?)</(?:th|td)>', 
                row_html, re.DOTALL | re.IGNORECASE
            )
            # 清理单元格内容
            clean_cells = []
            for cell in cells:
                # 去除HTML标签
                cell_text = re.sub(r'<[^>]+>', '', cell).strip()
                # 去除多余空白
                cell_text = re.sub(r'\s+', ' ', cell_text)
                clean_cells.append(cell_text)
            
            if clean_cells:
                parsed_rows.append(clean_cells)
        
        if not parsed_rows:
            continue
        
        # 判断第一行是否为表头
        # 如果原HTML中有<th>标签，则第一行为表头
        has_header = bool(re.search(r'<th[^>]*>', rows[0], re.IGNORECASE))
        
        # 转换为文本
        text_lines = []
        
        if has_header and len(parsed_rows) > 1:
            headers = parsed_rows[0]
            data_rows = parsed_rows[1:]
            
            for data_row in data_rows:
                row_parts = []
                for i, val in enumerate(data_row):
                    if val:  # 跳过空值
                        header = headers[i] if i < len(headers) else f"列{i+1}"
                        row_parts.append(f"{header}: {val}")
                if row_parts:
                    text_lines.append("；".join(row_parts))
        else:
            # 无表头，直接列出
            for row in parsed_rows:
                non_empty = [c for c in row if c]
                if non_empty:
                    text_lines.append("；".join(non_empty))
        
        table_text = "\n".join(text_lines)
        result = result.replace(table_html, table_text)
    
    return result


def clean_image_references(text: str) -> str:
    """
    清理Markdown图片引用，保留图片说明文字。
    
    处理模式：
    - ![alt_text](url) → 保留alt_text（如果有）
    - 单独的图片说明行（如"图 5.1 xxx"）→ 保留
    - 纯图片引用无alt_text → 移除
    """
    # 替换 ![alt](url) 为 alt_text（如果有意义的话）
    def replace_img(match):
        alt_text = match.group(1).strip()
        if alt_text and len(alt_text) > 2:
            return f"[图片: {alt_text}]"
        return ""
    
    result = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', replace_img, text)
    
    # 清理产生的多余空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result


def is_reference_text(text: str) -> bool:
    """
    判断文本是否为参考文献内容。
    
    检测模式：
    - 中文参考文献: 作者. 年份. 标题[J/M/D/C]. 期刊/出版社
    - 英文参考文献: Author. Year. Title [J/M/C]. Journal
    - 混合格式
    
    判断标准：如果文本中参考文献条目占比超过50%，则判定为参考文献文本。
    """
    if not text or len(text.strip()) < 20:
        return False
    
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False
    
    # 参考文献条目的特征模式
    ref_patterns = [
        r'\[\s*[JMDC]\s*\]',                          # [J]. [M]. [D]. [C]
        r'\d{4}\s*[\.．。]\s*\S+.*\[\s*[JMDC]\s*\]',  # 年份...标题[J]
        r'^\w+\s+\w+.*\d{4}.*\[\s*[JMDC]\s*\]',      # 英文作者...年份...[J]
        r'^\S+[，,]\s*\S+.*\d{4}.*\[\s*[JMDC]\s*\]',  # 中文作者，...年份...[J]
        r'\d{4}\.\s*\S+.*(?:Journal|Proceedings|Conference)', # 英文期刊
        r'(?:出版社|Press|Publisher)',                   # 出版社
    ]
    
    ref_line_count = 0
    for line in lines:
        for pattern in ref_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                ref_line_count += 1
                break
    
    # 如果超过50%的行匹配参考文献模式，则判定为参考文献
    ratio = ref_line_count / len(lines) if lines else 0
    return ratio > 0.5


def preprocess_md_text(text: str) -> str:
    """
    Markdown文本综合预处理管线。
    
    处理顺序：
    1. 基础清洗（多余空行、空白字符）
    2. HTML表格转文本
    3. 图片引用清理
    4. 最终清洗
    
    Args:
        text: 原始Markdown文本
        
    Returns:
        预处理后的文本
    """
    # Step 1: 基础清洗
    text = clean_text(text)
    
    # Step 2: HTML表格转文本
    if '<table' in text.lower():
        text = html_table_to_text(text)
    
    # Step 3: 图片引用清理
    if '![' in text:
        text = clean_image_references(text)
    
    # Step 4: 最终清洗
    text = clean_text(text)
    
    return text


# ============================================================
# 内容类型检测
# ============================================================

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """简单的关键词提取（基于词频，中文场景）"""
    clean = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
    words = [w for w in clean.split() if len(w) >= 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def is_definition_text(text: str) -> bool:
    """判断文本是否为定义型内容"""
    definition_patterns = [
        r'是指', r'定义为', r'被定义为', r'是一种', r'称为', r'即',
        r'概念', r'术语', r'是.*的总称', r'指的是', r'简称',
        r'英文.*为', r'缩写为', r'全称为'
    ]
    return any(re.search(p, text) for p in definition_patterns)


def is_case_text(text: str) -> bool:
    """判断文本是否为案例型内容（故障分析、事故案例等）"""
    case_patterns = [
        r'故障', r'事故', r'案例', r'异常', r'报错', r'失效',
        r'由于.*导致', r'因此', r'原因.*分析', r'排查', r'诊断',
        r'根本原因', r'失败.*原因', r'问题.*分析', r'处置',
        r'应急', r'抢救', r'修复', r'排除'
    ]
    return sum(1 for p in case_patterns if re.search(p, text)) >= 2


def is_process_text(text: str) -> bool:
    """判断文本是否为过程型内容（步骤、流程等）"""
    process_patterns = [
        r'步骤[一二三四五六七八九十\d]', r'第[一二三四五六七八九十\d]步',
        r'首先.*然后', r'流程', r'工序', r'程序',
        r'操作.*方法', r'实施.*步骤', r'\d+[\.、）\)].*\n.*\d+[\.、）\)]',
        r'阶段[一二三四五六七八九十\d]'
    ]
    return any(re.search(p, text) for p in process_patterns)


def is_data_text(text: str) -> bool:
    """
    判断文本是否为数据/参数密集型内容（v2.1新增）
    包含大量数值、单位、技术指标等
    """
    # 检测带单位的数值
    numeric_patterns = [
        r'\d+\.?\d*\s*(?:kg|km|m|cm|mm|Hz|MHz|GHz|W|kW|MW|V|A|dB|dBm|dBW|bps|kbps|Mbps)',
        r'\d+\.?\d*\s*(?:℃|°C|K|Pa|MPa|kPa|bar|N|kN|s|ms|μs|ns)',
        r'\d+\.?\d*\s*(?:bit|byte|KB|MB|GB|TB)',
        r'[≥≤><±]\s*\d+',
        r'\d+\s*[×x]\s*\d+',
    ]
    hit_count = sum(
        len(re.findall(p, text, re.IGNORECASE)) 
        for p in numeric_patterns
    )
    return hit_count >= 3


def detect_content_type(text: str) -> str:
    """
    基于规则的内容类型初步检测。
    返回: definition / case / process / data / reference / description
    v2.1: 增加data和reference类型
    """
    if is_reference_text(text):
        return "reference"
    elif is_case_text(text):
        return "case"
    elif is_definition_text(text):
        return "definition"
    elif is_process_text(text):
        return "process"
    elif is_data_text(text):
        return "data"
    else:
        return "description"


# ============================================================
# JSON处理
# ============================================================

def safe_json_parse(text: str) -> Optional[Dict]:
    """安全的JSON解析，支持从Markdown代码块中提取JSON"""
    if not text:
        return None
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试从```json ... ```代码块中提取
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取最外层的JSON对象
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


# ============================================================
# 文件IO
# ============================================================

def generate_chunk_id(book_name: str, chapter: str, section: str, index: int) -> str:
    """生成chunk的唯一标识符"""
    chapter_clean = re.sub(r'[^\w]', '', chapter)[:20]
    section_clean = re.sub(r'[^\w]', '', section)[:20]
    return f"{book_name}_ch{chapter_clean}_sec{section_clean}_p{index:04d}"


def save_jsonl(data: List[Dict], path: str):
    """保存为JSONL格式"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL格式文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
