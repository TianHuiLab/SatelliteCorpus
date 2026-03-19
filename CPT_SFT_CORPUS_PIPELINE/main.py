#!/usr/bin/env python3
"""
航天领域高质量微调语料生产Pipeline v2.0 - 主控程序
===================================================
知识密度过滤版：先判断后生成，严格过滤低价值内容。

使用方法：
    # 基础用法：处理单个MD文件
    python main.py --input data/book.md --output output/

    # 处理整个目录
    python main.py --input data/ --output output/

    # 使用配置文件
    python main.py --config config.json

    # 仅生成CPT语料（不调用API）
    python main.py --input data/book.md --output output/ --mode cpt

    # 生成SFT语料（需要API Key）
    python main.py --input data/book.md --output output/ --mode sft --api-key sk-xxx

    # 完整模式（CPT + SFT）
    python main.py --input data/book.md --output output/ --mode full --api-key sk-xxx

    # 冷启动测试（仅处理前N个chunk）
    python main.py --input data/book.md --output output/ --mode sft --cold-start 3

    # 禁用LLM过滤（仅使用规则引擎过滤）
    python main.py --input data/book.md --mode sft --no-llm-filter --api-key sk-xxx

    # 禁用所有过滤（所有chunk都生成语料）
    python main.py --input data/book.md --mode sft --no-filter --api-key sk-xxx
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, ChunkerConfig, LLMConfig, SFTConfig, FilterConfig, PROVIDER_CONFIGS
from src.data_loader import MarkdownLoader, Document
from src.chunker import SemanticChunker, Chunk
from src.cpt_generator import CPTGenerator
from src.sft_generator import SFTGenerator
from src.knowledge_filter import KnowledgeDensityFilter
from src.utils import setup_logger, get_timestamp, save_jsonl


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="航天领域高质量微调语料生产Pipeline v3.1（合并调用+多线程版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --input data/book.md --output output/ --mode cpt
  python main.py --input data/ --output output/ --mode full --api-key sk-xxx
  python main.py --config pipeline_config.json
  python main.py --input data/book.md --mode sft --no-llm-filter --api-key sk-xxx
  python main.py --input data/book.md --mode sft --api-key sk-xxx --workers 4
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入MD文件路径或目录路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="输出目录路径 (默认: ./output)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["cpt", "sft", "full"],
        default="full",
        help="运行模式: cpt(仅CPT) / sft(仅SFT) / full(CPT+SFT) (默认: full)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=list(PROVIDER_CONFIGS.keys()),
        default=None,
        help=(
            "API提供商预设（自动配置 base-url 和默认 model）: "
            "dashscope(Qwen，默认) / deepseek / openai。"
            "显式指定 --model 或 --base-url 可覆盖提供商默认值。"
        )
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help=(
            "LLM API Key。也可通过环境变量设置: "
            "DASHSCOPE_API_KEY（DashScope/Qwen）/ "
            "DEEPSEEK_API_KEY（DeepSeek）/ "
            "OPENAI_API_KEY（OpenAI）"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM模型名称（不指定时由 --provider 决定默认值，如 qwen3-max / deepseek-chat）"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API Base URL（不指定时由 --provider 决定默认值）"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="aerospace",
        help="领域标识 (默认: aerospace)"
    )
    parser.add_argument(
        "--cold-start",
        type=int,
        default=0,
        help="冷启动模式：仅处理前N个chunk (默认: 0=处理全部)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["sharegpt", "alpaca", "messages"],
        default="sharegpt",
        help="SFT输出格式 (默认: sharegpt)"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="最小分块字符数 (默认: 200)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="最大分块字符数 (默认: 800)"
    )
    parser.add_argument(
        "--glossary",
        type=str,
        help="领域术语表文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    # v2.0新增：知识密度过滤选项
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="禁用知识密度过滤（所有chunk都生成语料）"
    )
    parser.add_argument(
        "--no-llm-filter",
        action="store_true",
        help="禁用LLM过滤（仅使用规则引擎过滤，节省API调用）"
    )
    # v3.1新增：并发线程数
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数（默认: 1=顺序执行；建议DashScope用3-5，需符合API限流）"
    )

    return parser.parse_args()


class Pipeline:
    """
    微调语料生产Pipeline v2.0 主控类
    
    流程：
    1. 加载配置
    2. 读取MD文件 → 解析文档结构
    3. 语义分块 → 生成Chunk列表
    4. 生成CPT语料（可选）
    5. 知识密度过滤 → 拦截低价值内容（v2.0新增）
    6. 调用LLM API动态生成SFT语料（可选）
    7. 质量过滤与统计
    8. 保存输出文件与报告
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = get_timestamp()
        
        # 创建输出目录
        self.output_dir = os.path.join(config.output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化日志
        log_file = os.path.join(self.output_dir, "pipeline.log")
        self.logger = setup_logger("Pipeline", config.log_level, log_file)
        
        # 初始化各模块
        self.loader = MarkdownLoader(
            heading_levels=config.chunker.heading_levels
        )
        self.chunker = SemanticChunker(
            min_chunk_chars=config.chunker.min_chunk_chars,
            max_chunk_chars=config.chunker.max_chunk_chars,
            overlap_chars=config.chunker.overlap_chars,
            preserve_context_anchor=config.chunker.preserve_context_anchor
        )
        self.cpt_gen = CPTGenerator(
            domain=config.domain,
            add_metadata_header=config.cpt.add_metadata_header,
            separator=config.cpt.separator,
            output_format=config.cpt.output_format
        )
        self.sft_gen = None  # 延迟初始化（需要API Key）
        
        # 全局统计
        self.report = {
            "timestamp": self.timestamp,
            "pipeline_version": "3.1",
            "config": {},
            "documents": [],
            "chunking": {},
            "cpt": {},
            "filter": {},
            "sft": {},
            "quality": {}
        }
    
    def _init_sft_generator(self):
        """初始化SFT生成器（需要API Key）"""
        if not self.config.llm.api_key:
            provider = self.config.llm.provider
            provider_cfg = PROVIDER_CONFIGS.get(provider, {})
            env_key = provider_cfg.get("env_key", "DASHSCOPE_API_KEY")
            raise ValueError(
                f"SFT模式需要API Key。请通过 --api-key 参数或 "
                f"环境变量 {env_key} 设置（当前提供商: {provider}）。\n"
                f"其他支持的环境变量: "
                + " / ".join(p["env_key"] for p in PROVIDER_CONFIGS.values())
            )
        
        # 加载术语表
        glossary = {}
        if self.config.sft.domain_glossary_path:
            try:
                with open(self.config.sft.domain_glossary_path, 'r', encoding='utf-8') as f:
                    glossary = json.load(f)
                self.logger.info(f"已加载术语表: {len(glossary)} 个术语")
            except Exception as e:
                self.logger.warning(f"术语表加载失败: {e}")
        
        self.sft_gen = SFTGenerator(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            model=self.config.llm.model,
            provider=self.config.llm.provider,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            top_p=self.config.llm.top_p,
            timeout=self.config.llm.timeout,
            max_retries=self.config.llm.max_retries,
            retry_delay=self.config.llm.retry_delay,
            batch_delay=self.config.llm.batch_delay,
            domain=self.config.domain,
            confidence_threshold=self.config.sft.confidence_threshold,
            max_samples_per_chunk=self.config.sft.max_samples_per_chunk,
            enable_negative_examples=self.config.sft.enable_negative_examples,
            enable_thinking=self.config.llm.enable_thinking,
            enable_llm_filter=self.config.filter.enable_llm_filter,
            glossary=glossary,
            max_workers=self.config.llm.max_workers,   # v3.1
        )
    
    def run(self, mode: str = "full", cold_start: int = 0) -> Dict:
        """
        执行Pipeline
        
        Args:
            mode: 运行模式 (cpt / sft / full)
            cold_start: 冷启动模式下处理的chunk数量 (0=全部)
            
        Returns:
            Dict: 运行报告
        """
        start_time = time.time()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Pipeline v3.1 启动 | 模式={mode} | 时间={self.timestamp}")
        self.logger.info(f"提供商: {self.config.llm.provider} | 模型: {self.config.llm.model}")
        self.logger.info(f"知识密度过滤: {'启用' if self.config.filter.enable else '禁用'}")
        self.logger.info(f"并发线程数: {self.config.llm.max_workers}")
        self.logger.info(f"{'='*60}")
        
        # ========== Step 1: 加载文档 ==========
        self.logger.info("[Step 1/7] 加载Markdown文档...")
        documents = self._load_documents()
        if not documents:
            self.logger.error("未找到任何文档，Pipeline终止。")
            return self.report
        
        # ========== Step 2: 语义分块 ==========
        self.logger.info("[Step 2/7] 执行语义分块...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            
            self.report["documents"].append({
                "filename": doc.filename,
                "book_name": doc.book_name,
                "total_chars": doc.total_chars,
                "total_sections": doc.total_sections,
                "chunks_count": len(chunks)
            })
        
        # 冷启动模式
        if cold_start > 0:
            self.logger.info(f"冷启动模式: 仅处理前 {cold_start} 个chunk")
            all_chunks = all_chunks[:cold_start]
        
        chunk_stats = self.chunker.get_statistics(all_chunks)
        self.report["chunking"] = chunk_stats
        self.logger.info(f"分块完成: {chunk_stats}")
        
        # 保存分块结果
        self._save_chunks(all_chunks)
        
        # ========== Step 3: 生成CPT语料 ==========
        if mode in ("cpt", "full"):
            self.logger.info("[Step 3/7] 生成CPT语料...")
            self._generate_cpt(all_chunks, documents)
        else:
            self.logger.info("[Step 3/7] 跳过CPT生成（mode=sft）")
        
        # ========== Step 4: 知识密度预过滤（独立阶段） ==========
        if mode in ("sft", "full"):
            if self.config.filter.enable:
                self.logger.info("[Step 4/7] 执行知识密度预过滤...")
                filtered_chunks, rejected_chunks, filter_details = self._run_knowledge_filter(all_chunks)
                
                # 保存过滤详情
                filter_detail_path = os.path.join(self.output_dir, "filter_details.jsonl")
                save_jsonl(filter_details, filter_detail_path)
                self.logger.info(f"过滤详情已保存: {filter_detail_path}")
                
                # 保存被过滤的chunk（便于审查）
                if rejected_chunks:
                    rejected_path = os.path.join(self.output_dir, "filtered_chunks.jsonl")
                    rejected_data = [
                        {"chunk_id": c.chunk_id, "section_title": c.section_title, 
                         "content": c.content[:200], "char_count": c.char_count}
                        for c in rejected_chunks
                    ]
                    save_jsonl(rejected_data, rejected_path)
                    self.logger.info(f"被过滤chunk已保存: {rejected_path}")
                
                # 更新报告
                self.report["filter"] = {
                    "total_chunks": len(all_chunks),
                    "passed_chunks": len(filtered_chunks),
                    "rejected_chunks": len(rejected_chunks),
                    "filter_rate": f"{len(rejected_chunks)/len(all_chunks)*100:.1f}%" if all_chunks else "0%"
                }
                
                # 使用过滤后的chunk进行SFT生成
                sft_chunks = filtered_chunks
            else:
                self.logger.info("[Step 4/7] 知识密度过滤已禁用，使用全部chunk")
                sft_chunks = all_chunks
                self.report["filter"] = {"status": "disabled"}
            
            # ========== Step 5: 生成SFT语料 ==========
            self.logger.info("[Step 5/7] 生成SFT语料...")
            self._init_sft_generator()
            self._generate_sft(sft_chunks)
        else:
            self.logger.info("[Step 4/7] 跳过知识密度过滤（mode=cpt）")
            self.logger.info("[Step 5/7] 跳过SFT生成（mode=cpt）")
        
        # ========== Step 6: 质量统计 ==========
        self.logger.info("[Step 6/7] 生成质量报告...")
        
        # ========== Step 7: 保存报告 ==========
        self.logger.info("[Step 7/7] 保存运行报告...")
        
        elapsed = time.time() - start_time
        self.report["elapsed_seconds"] = round(elapsed, 2)
        self.report["config"] = {
            "mode": mode,
            "cold_start": cold_start,
            "domain": self.config.domain,
            "provider": self.config.llm.provider,
            "model": self.config.llm.model,
            "max_workers": self.config.llm.max_workers,
            "filter_enabled": self.config.filter.enable,
            "llm_filter_enabled": self.config.filter.enable_llm_filter
        }
        
        self._save_report()
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Pipeline v3.1 完成 | 耗时={elapsed:.1f}秒")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"{'='*60}")
        
        return self.report
    
    def _load_documents(self) -> List[Document]:
        """加载文档"""
        input_path = self.config.input_dir
        
        if os.path.isfile(input_path):
            doc = self.loader.load_file(input_path)
            if self.config.book_name:
                doc.book_name = self.config.book_name
            return [doc]
        elif os.path.isdir(input_path):
            return self.loader.load_directory(input_path)
        else:
            self.logger.error(f"输入路径不存在: {input_path}")
            return []
    
    def _save_chunks(self, chunks: List[Chunk]):
        """保存分块结果"""
        chunks_path = os.path.join(self.output_dir, "chunks.jsonl")
        data = []
        for c in chunks:
            data.append({
                "chunk_id": c.chunk_id,
                "content": c.content,
                "context_anchor": c.context_anchor,
                "chapter_title": c.chapter_title,
                "section_title": c.section_title,
                "parent_titles": c.parent_titles,
                "type_hint": c.type_hint,
                "char_count": c.char_count,
                "chunk_index": c.chunk_index,
                "prev_summary": c.prev_summary,
                "content_hash": c.content_hash,
                "is_reference": getattr(c, 'is_reference', False),
                "has_formula": getattr(c, 'has_formula', False),
                "has_table_data": getattr(c, 'has_table_data', False)
            })
        save_jsonl(data, chunks_path)
        self.logger.info(f"分块结果已保存: {chunks_path}")
    
    def _generate_cpt(self, chunks: List[Chunk], documents: List[Document]):
        """生成CPT语料"""
        for doc in documents:
            doc_chunks = [c for c in chunks if c.chapter_title or True]
            
            # 基础CPT
            cpt_samples = self.cpt_gen.generate(doc_chunks, doc.book_name)
            cpt_path = os.path.join(self.output_dir, f"cpt_{doc.book_name}.jsonl")
            self.cpt_gen.save(cpt_samples, cpt_path)
            
            # 上下文CPT
            ctx_samples = self.cpt_gen.generate_contextual(
                doc_chunks, doc.book_name, window_size=2
            )
            ctx_path = os.path.join(self.output_dir, f"cpt_contextual_{doc.book_name}.jsonl")
            self.cpt_gen.save(ctx_samples, ctx_path)
            
            cpt_stats = self.cpt_gen.get_statistics(cpt_samples)
            self.report["cpt"] = cpt_stats
    
    def _run_knowledge_filter(self, chunks: List[Chunk]):
        """
        执行知识密度预过滤（独立阶段，不消耗LLM API）
        
        使用规则引擎对所有chunk进行预过滤，
        LLM过滤在SFT生成阶段内部执行。
        """
        kf = KnowledgeDensityFilter(
            min_chars=self.config.filter.min_valid_chars,
            high_value_threshold=self.config.filter.high_value_threshold,
            enable_llm_filter=False,  # 预过滤阶段仅用规则引擎，节省API
            domain=self.config.domain
        )
        
        passed, rejected, details = kf.filter_chunks(chunks)
        
        filter_stats = kf.get_statistics()
        self.logger.info(
            f"预过滤完成: 通过={len(passed)}, 过滤={len(rejected)}, "
            f"过滤率={filter_stats.get('filter_rate', 0):.1f}%"
        )
        
        return passed, rejected, details
    
    def _generate_sft(self, chunks: List[Chunk]):
        """生成SFT语料（已经过知识密度预过滤的chunks）"""
        def progress_callback(current, total, chunk_id):
            if current % 5 == 0 or current == total:
                self.logger.info(f"SFT进度: {current}/{total} ({current/total*100:.1f}%)")
        
        # 批量处理
        good_samples, low_conf_samples, diagnoses = self.sft_gen.process_chunks(
            chunks, progress_callback
        )
        
        # 保存合格样本
        sft_format = self.config.sft.output_format
        sft_path = os.path.join(self.output_dir, f"sft_train.{sft_format}.jsonl")
        self.sft_gen.save_samples(good_samples, sft_path, format=sft_format)
        
        # 保存低置信度样本（人工审核池）
        if low_conf_samples:
            review_path = os.path.join(self.output_dir, "sft_review_pool.jsonl")
            self.sft_gen.save_samples(low_conf_samples, review_path, format=sft_format)
            self.logger.info(f"低置信度样本已保存: {review_path} ({len(low_conf_samples)} 条)")
        
        # 保存诊断结果
        diag_path = os.path.join(self.output_dir, "diagnoses.jsonl")
        save_jsonl(diagnoses, diag_path)
        
        # 同时输出Alpaca格式
        alpaca_path = os.path.join(self.output_dir, "sft_train.alpaca.jsonl")
        self.sft_gen.save_samples(good_samples, alpaca_path, format="alpaca")
        
        # 更新报告
        sft_stats = self.sft_gen.get_statistics()
        sft_stats["good_samples"] = len(good_samples)
        sft_stats["low_confidence_samples"] = len(low_conf_samples)
        self.report["sft"] = sft_stats
    
    def _save_report(self):
        """保存运行报告"""
        report_path = os.path.join(self.output_dir, "pipeline_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"运行报告已保存: {report_path}")
        
        # 生成可读的Markdown报告
        md_report = self._generate_md_report()
        md_path = os.path.join(self.output_dir, "pipeline_report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        self.logger.info(f"Markdown报告已保存: {md_path}")
    
    def _generate_md_report(self) -> str:
        """生成Markdown格式的运行报告"""
        r = self.report
        
        lines = [
            f"# Pipeline v2.2 运行报告（知识密度过滤版）",
            f"",
            f"**运行时间**: {r.get('timestamp', '')}",
            f"**Pipeline版本**: {r.get('pipeline_version', '2.0')}",
            f"**运行模式**: {r.get('config', {}).get('mode', '')}",
            f"**领域**: {r.get('config', {}).get('domain', '')}",
            f"**模型**: {r.get('config', {}).get('model', '')}",
            f"**知识密度过滤**: {'启用' if r.get('config', {}).get('filter_enabled', True) else '禁用'}",
            f"**总耗时**: {r.get('elapsed_seconds', 0):.1f} 秒",
            f"",
            f"## 文档信息",
            f"",
            f"| 文件名 | 书籍名称 | 总字符数 | 章节数 | 分块数 |",
            f"|--------|----------|----------|--------|--------|",
        ]
        
        for doc in r.get("documents", []):
            lines.append(
                f"| {doc['filename']} | {doc['book_name']} | "
                f"{doc['total_chars']:,} | {doc['total_sections']} | "
                f"{doc['chunks_count']} |"
            )
        
        # 分块统计
        cs = r.get("chunking", {})
        if cs:
            lines.extend([
                f"",
                f"## 分块统计",
                f"",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 总分块数 | {cs.get('total_chunks', 0)} |",
                f"| 总字符数 | {cs.get('total_chars', 0):,} |",
                f"| 平均字符数 | {cs.get('avg_chars', 0):.0f} |",
                f"| 最小字符数 | {cs.get('min_chars', 0)} |",
                f"| 最大字符数 | {cs.get('max_chars', 0)} |",
                f"| 章节数 | {cs.get('chapters', 0)} |",
            ])
            
            type_dist = cs.get("type_distribution", {})
            if type_dist:
                lines.extend([
                    f"",
                    f"### 内容类型分布（规则引擎初判）",
                    f"",
                    f"| 类型 | 数量 | 占比 |",
                    f"|------|------|------|",
                ])
                total = sum(type_dist.values())
                for t, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
                    pct = cnt / total * 100 if total > 0 else 0
                    lines.append(f"| {t} | {cnt} | {pct:.1f}% |")
        
        # 知识密度过滤统计（v2.0新增）
        ft = r.get("filter", {})
        if ft and ft.get("status") != "disabled":
            lines.extend([
                f"",
                f"## 知识密度过滤统计",
                f"",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 输入chunk数 | {ft.get('total_chunks', 0)} |",
                f"| 通过chunk数 | {ft.get('passed_chunks', 0)} |",
                f"| 过滤chunk数 | {ft.get('rejected_chunks', 0)} |",
                f"| 过滤率 | {ft.get('filter_rate', '0%')} |",
            ])
        
        # CPT统计
        cpt = r.get("cpt", {})
        if cpt:
            lines.extend([
                f"",
                f"## CPT语料统计",
                f"",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 总样本数 | {cpt.get('total_samples', 0)} |",
                f"| 总字符数 | {cpt.get('total_chars', 0):,} |",
                f"| 平均字符数 | {cpt.get('avg_chars', 0):.0f} |",
            ])
        
        # SFT统计
        sft = r.get("sft", {})
        if sft:
            lines.extend([
                f"",
                f"## SFT语料统计",
                f"",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 输入chunk数 | {sft.get('total_chunks_input', 0)} |",
                f"| 二次过滤数 | {sft.get('chunks_filtered', 0)} |",
                f"| 实际生成chunk数 | {sft.get('chunks_passed', 0)} |",
                f"| 合格样本数 | {sft.get('good_samples', 0)} |",
                f"| 低置信度样本数 | {sft.get('low_confidence_samples', 0)} |",
                f"| 总API调用次数 | {sft.get('total_api_calls', 0)} |",
                f"| 总Token消耗 | {sft.get('total_tokens_used', 0):,} |",
                f"| 成功生成数 | {sft.get('successful_generations', 0)} |",
                f"| 失败生成数 | {sft.get('failed_generations', 0)} |",
            ])
            
            type_counts = sft.get("type_counts", {})
            if type_counts:
                lines.extend([
                    f"",
                    f"### SFT语料类型分布",
                    f"",
                    f"| 类型 | 数量 | 说明 |",
                    f"|------|------|------|",
                ])
                type_desc = {
                    "fact_qa": "基础事实问答",
                    "comprehension": "理解型问题",
                    "reasoning_qa": "推理问题",
                    "summarization": "总结任务",
                    "information_extraction": "信息抽取任务",
                    "calculation": "计算问题",
                }
                for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                    desc = type_desc.get(t, t)
                    lines.append(f"| {t} | {cnt} | {desc} |")
        
        lines.extend([
            f"",
            f"---",
            f"*报告由航天领域微调语料生产Pipeline v2.2（知识密度过滤版）自动生成*",
        ])
        
        return "\n".join(lines)


def main():
    """主入口函数"""
    args = parse_args()
    
    # 加载或构建配置
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()
    
    # 命令行参数覆盖配置
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.domain:
        config.domain = args.domain
    if args.format:
        config.sft.output_format = args.format
    if args.min_chars:
        config.chunker.min_chunk_chars = args.min_chars
    if args.max_chars:
        config.chunker.max_chunk_chars = args.max_chars
    if args.glossary:
        config.sft.domain_glossary_path = args.glossary
    if args.log_level:
        config.log_level = args.log_level
    
    # ========== API 提供商配置（核心逻辑）==========
    # 优先级: --provider 推断的默认值 < --model/--base-url 显式指定 < --api-key 显式指定
    provider = args.provider or "dashscope"
    provider_cfg = PROVIDER_CONFIGS[provider]
    config.llm.provider = provider

    # base_url：显式指定 > provider 默认值
    config.llm.base_url = args.base_url or provider_cfg["base_url"]

    # model：显式指定 > provider 默认值
    config.llm.model = args.model or provider_cfg["default_model"]

    # api_key：CLI > provider对应env var > 其他provider env var
    if args.api_key:
        config.llm.api_key = args.api_key
    elif not config.llm.api_key:
        # 先查当前 provider 的 env var
        config.llm.api_key = os.getenv(provider_cfg["env_key"], "")
    if not config.llm.api_key:
        # fallback：遍历所有 provider 的 env var
        for _, pcfg in PROVIDER_CONFIGS.items():
            val = os.getenv(pcfg["env_key"], "")
            if val:
                config.llm.api_key = val
                break

    if args.provider:
        print(
            f"[Provider] {provider} | "
            f"model={config.llm.model} | "
            f"base_url={config.llm.base_url}"
        )
    
    # v2.0: 知识密度过滤选项
    if args.no_filter:
        config.filter.enable = False
        config.filter.enable_llm_filter = False
    elif args.no_llm_filter:
        config.filter.enable_llm_filter = False

    # v3.1: 并发线程数
    if args.workers and args.workers > 0:
        config.llm.max_workers = args.workers
    
    # 保存本次运行的配置
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 创建并运行Pipeline
    pipeline = Pipeline(config)
    
    # 保存配置快照
    config_snapshot_path = os.path.join(pipeline.output_dir, "config_snapshot.json")
    config.save(config_snapshot_path)
    
    report = pipeline.run(
        mode=args.mode,
        cold_start=args.cold_start
    )
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Pipeline v3.1 执行完成!")
    print(f"输出目录: {pipeline.output_dir}")
    print(f"耗时: {report.get('elapsed_seconds', 0):.1f} 秒")
    
    if "chunking" in report and report["chunking"]:
        print(f"总分块数: {report['chunking'].get('total_chunks', 0)}")
    
    ft = report.get("filter", {})
    if ft and ft.get("status") != "disabled":
        print(f"知识密度过滤: 通过={ft.get('passed_chunks', 0)}, 过滤={ft.get('rejected_chunks', 0)}")
    
    if "sft" in report and report["sft"]:
        print(f"SFT合格样本: {report['sft'].get('good_samples', 0)}")
        print(f"API调用次数: {report['sft'].get('total_api_calls', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
