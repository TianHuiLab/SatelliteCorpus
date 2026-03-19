import os
import re
import sys
import json
from typing import List, Dict
from openai import OpenAI

# 默认配置
DEFAULT_MODEL = "qwen3-max"  # 映射到环境中的 Qwen3-max 或类似模型
DEFAULT_CHUNK_SIZE = 3000      # 每个分块的大小

class QwenMDStructureFixer:
    """
    基于 LLM (Qwen3-max) 的 Markdown 结构智能修复工具。
    利用目录信息，在正文的正确位置智能补全或修正标题。
    """

    def __init__(self, base_url: str = None, model: str = DEFAULT_MODEL):
        # 优先使用环境变量中的配置
        self.api_key = "sk-ac6ac1cd572d4a43b135c206756758b9"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def extract_toc(self, content: str) -> str:
        """从文档中提取目录部分作为 LLM 的全局参考"""
        # 寻找目录部分的起始和结束
        toc_match = re.search(r'(#+\s*目录.*?\n)(.*?)(?=\n#|\Z)', content, re.S | re.I)
        if toc_match:
            return toc_match.group(0).strip()
        # 如果没找到显式的目录标记，尝试寻找类似目录的列表
        toc_lines = []
        lines = content.split('\n')
        for line in lines:
            if re.match(r'^(第\s*\d+\s*[章回]|(?:\d+(?:\.\d+)+))\s+', line.strip()):
                toc_lines.append(line.strip())
        return "\n".join(toc_lines[:100]) # 限制长度

    def fix_chunk(self, toc: str, chunk: str, prev_context: str = "") -> str:
        """调用 Qwen3-max 修复单个正文块"""
        system_prompt = """你是一个专业的学术文档结构修复专家。
你的任务是根据提供的【目录结构】，对【正文片段】进行智能标题补全和层级修正。

### 核心原则：
1. **查漏补缺**：仅在【正文片段】确实是目录中某个章节的起始位置，但缺失了对应标题时才插入标题。
2. **位置准确**：严禁在不相关的地方强行插入标题（例如第二章的内容中绝不能出现第一章的标题）。
3. **层级修正**：如果正文中已有标题但层级不对（如目录要求是二级 ##，正文只有文本或三级 ###），请将其修正为标准 Markdown 格式。
4. **内容保真**：严禁修改、删除或增加正文中的任何学术内容、公式、数据或标点。仅针对标题行进行操作。
5. **去噪**：如果识别到的标题行中包含目录中的页码（如 …… 19），请将其剔除。

### 输出要求：
直接输出修复后的 Markdown 正文，不要包含任何解释、代码块标记或 Markdown 以外的内容。"""

        user_prompt = f"""【目录结构】：
{toc}

【前文语境（仅供参考位置，请勿重复输出）】：
{prev_context[-500:] if prev_context else "文档开头"}

【待修复的正文片段】：
{chunk}

请根据目录信息，对上述正文片段进行标题补全或修正："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # 保持稳定性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API 调用失败: {e}")
            return chunk

    def process_file(self, input_path: str, output_path: str = None):
        """处理文件主逻辑"""
        if not os.path.exists(input_path):
            print(f"错误: 文件 {input_path} 不存在")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            full_content = f.read()

        print(f"[*] 正在分析目录结构...")
        toc = self.extract_toc(full_content)
        if not toc:
            print("[!] 未能提取到有效目录，修复效果可能受限。")

        # 分块处理正文（避开目录部分以节省 token 并防止自修改）
        # 简单处理：找到目录后的第一个正文行
        toc_end_pos = full_content.find(toc) + len(toc) if toc in full_content else 0
        pre_content = full_content[:toc_end_pos]
        main_content = full_content[toc_end_pos:]

        chunks = [main_content[i:i + DEFAULT_CHUNK_SIZE] for i in range(0, len(main_content), DEFAULT_CHUNK_SIZE)]
        print(f"[*] 正文已切分为 {len(chunks)} 个分块，开始智能修复...")

        fixed_main_content = ""
        prev_context = ""

        for i, chunk in enumerate(chunks):
            print(f"    [进度] 正在处理第 {i+1}/{len(chunks)} 个分块...")
            fixed_chunk = self.fix_chunk(toc, chunk, prev_context)
            fixed_main_content += fixed_chunk + "\n"
            prev_context = fixed_chunk

        final_content = pre_content + "\n" + fixed_main_content

        if not output_path:
            output_path = input_path.replace('.md', '_qwen_fixed.md')
            if output_path == input_path:
                output_path += ".qwen"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"[√] 修复完成！结果已保存至: {output_path}")

if __name__ == "__main__":

    input_file = "F:/SatelliteCorpus/md_books/高分辨率卫星任务规划技术.md"
    output_file = "F:/SatelliteCorpus/md_books/高分辨率卫星任务规划技术_fixed.md"

    # 初始化并运行
    fixer = QwenMDStructureFixer()
    fixer.process_file(input_file, output_file)
