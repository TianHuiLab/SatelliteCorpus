"""
高质量微调语料生产Pipeline - 全局配置 v3.1
=============================================
v3.1: 更新语料类型分布；新增多线程并发支持；新增多提供商API预设配置。
支持通过JSON配置文件或环境变量覆盖默认参数。
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ============================================================
# 支持的API提供商预设配置
# ============================================================
PROVIDER_CONFIGS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3-max",
        "env_key": "DASHSCOPE_API_KEY",
        "description": "阿里云 DashScope（Qwen 系列）",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "description": "DeepSeek（deepseek-chat / deepseek-reasoner）",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI（GPT-4o 等）",
    },
}


@dataclass
class ChunkerConfig:
    """分块器配置"""
    min_chunk_chars: int = 200          # 最小分块字符数
    max_chunk_chars: int = 800          # 最大分块字符数
    overlap_chars: int = 50             # 分块重叠字符数（保证上下文连贯）
    heading_levels: list = field(default_factory=lambda: [1, 2, 3, 4])  # 识别的标题层级
    preserve_context_anchor: bool = True  # 是否保留章节标题作为上下文锚点


@dataclass
class CPTConfig:
    """CPT语料生成配置"""
    add_metadata_header: bool = True     # 是否在CPT语料中添加元数据头
    separator: str = "\n\n"              # CPT语料块之间的分隔符
    output_format: str = "jsonl"         # 输出格式: jsonl / txt


@dataclass
class FilterConfig:
    """知识密度过滤配置（v2.0新增）"""
    enable: bool = True                  # 是否启用知识密度过滤
    min_valid_chars: int = 30            # 最小有效字符数，低于此值直接过滤
    high_value_threshold: int = 2        # 高价值关键词命中数阈值
    enable_llm_filter: bool = True       # 是否启用LLM精细过滤（规则引擎无法确定时）
    filter_sections: list = field(default_factory=lambda: [
        "前言", "序言", "目录", "致谢", "作者介绍",
        "出版说明", "版权", "参考文献", "索引"
    ])


@dataclass
class LLMConfig:
    """LLM API配置（v3.1: 支持多提供商）"""
    provider: str = "dashscope"          # API提供商: dashscope / deepseek / openai
    model: str = "qwen3-max"             # 模型名称（由provider自动推断或手动指定）
    api_key: str = ""                    # API Key（优先从对应provider的环境变量读取）
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API Base URL
    temperature: float = 0.3             # 生成温度（低温保证事实一致性）
    max_tokens: int = 4096               # 最大生成token数
    top_p: float = 0.9                   # Top-p采样
    timeout: int = 120                   # API请求超时时间（秒）
    max_retries: int = 3                 # 最大重试次数
    retry_delay: float = 2.0             # 重试间隔（秒）
    batch_delay: float = 0.5             # 批次间延迟（秒），顺序模式下避免限流
    enable_thinking: bool = False        # 是否启用思考模式（仅 dashscope/Qwen3 支持）
    max_workers: int = 1                 # v3.1新增：并发线程数（1=顺序, >1=并行）

    def __post_init__(self):
        # 若 provider 不是 dashscope 且用的仍是 dashscope 默认值，自动更新到对应 provider 的默认值
        if self.provider in PROVIDER_CONFIGS and self.provider != "dashscope":
            pcfg = PROVIDER_CONFIGS[self.provider]
            if self.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1":
                self.base_url = pcfg["base_url"]
            if self.model == "qwen3-max":
                self.model = pcfg["default_model"]

        if not self.api_key:
            # 按 provider 优先读取对应的环境变量
            provider_cfg = PROVIDER_CONFIGS.get(self.provider, {})
            env_key = provider_cfg.get("env_key", "")
            if env_key:
                self.api_key = os.getenv(env_key, "")
            # fallback：遍历所有 provider 的 env var
            if not self.api_key:
                for _, pcfg in PROVIDER_CONFIGS.items():
                    val = os.getenv(pcfg.get("env_key", ""), "")
                    if val:
                        self.api_key = val
                        break


@dataclass
class SFTConfig:
    """SFT语料生成配置（v3.0: 更新为新6种语料类型）"""
    output_format: str = "sharegpt"      # 输出格式: sharegpt / alpaca / messages
    confidence_threshold: float = 0.7    # 置信度阈值，低于此值进入人工审核池
    max_samples_per_chunk: int = 5       # 每个chunk最多生成的样本数
    enable_negative_examples: bool = True  # 是否在Prompt中加入负面示例
    domain_glossary_path: str = ""       # 领域术语表路径（可选）
    # 完整6种语料类型的目标分布（与sft_generator.py的VALID_TYPES保持一致）
    target_type_distribution: dict = field(default_factory=lambda: {
        "fact_qa": 0.25,                 # 基础事实问答
        "comprehension": 0.15,           # 理解型问题
        "reasoning_qa": 0.25,            # 推理问题（含原causal_reasoning+cot_reasoning）
        "summarization": 0.15,           # 总结任务
        "information_extraction": 0.15,  # 信息抽取任务
        "calculation": 0.05              # 计算问题
    })


@dataclass
class QualityConfig:
    """质量控制配置"""
    enable_json_validation: bool = True
    enable_source_check: bool = True
    min_answer_length: int = 20
    max_answer_length: int = 2000
    similarity_threshold: float = 0.75
    enable_dedup: bool = True


@dataclass
class PipelineConfig:
    """Pipeline总配置"""
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    cpt: CPTConfig = field(default_factory=CPTConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    input_dir: str = "./data"
    output_dir: str = "./output"
    log_level: str = "INFO"
    random_seed: int = 42
    domain: str = "aerospace"
    book_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = cls()
        if 'chunker' in data:
            config.chunker = ChunkerConfig(**data['chunker'])
        if 'cpt' in data:
            config.cpt = CPTConfig(**data['cpt'])
        if 'filter' in data:
            config.filter = FilterConfig(**data['filter'])
        if 'llm' in data:
            config.llm = LLMConfig(**data['llm'])
        if 'sft' in data:
            config.sft = SFTConfig(**data['sft'])
        if 'quality' in data:
            config.quality = QualityConfig(**data['quality'])
        for key in ['input_dir', 'output_dir', 'log_level', 'random_seed', 'domain', 'book_name']:
            if key in data:
                setattr(config, key, data[key])
        return config
