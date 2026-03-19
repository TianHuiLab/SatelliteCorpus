import re
import difflib

class MarkdownTitleFixer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.standard_titles = []
        self.content_lines = []
        self.fixed_content = []

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content_lines = f.readlines()

    def extract_toc(self):
        """
        第一阶段：提取目录作为真值表
        识别 ## 目录 之后直到下一个大标题出现前的所有行
        """
        toc_started = False
        for line in self.content_lines:
            clean_line = line.strip()
            if "目录" in clean_line and "##" in clean_line:
                toc_started = True
                continue
            if toc_started:
                # 假设目录项通常不以 # 开头，或者到下一个正式章节结束
                if clean_line.startswith('##') and "目录" not in clean_line:
                    break
                if clean_line:
                    # 移除可能的页码（如 ... 120）
                    title_only = re.sub(r'\s*\.+\s*\d+$', '', clean_line)
                    self.standard_titles.append(title_only)
        print(f"成功提取目录项: {len(self.standard_titles)} 条")

    def is_similar(self, text, threshold=0.85):
        """使用模糊匹配判断文本是否在目录清单中"""
        for std_title in self.standard_titles:
            # 去除空格后比对
            ratio = difflib.SequenceMatcher(None, text.replace(" ", ""), std_title.replace(" ", "")).ratio()
            if ratio >= threshold:
                return True, std_title
        return False, None

    def fix_titles(self):
        """
        第二阶段：遍历正文，修复类型A（断行）和类型B（缺失标识）
        """
        i = 0
        n = len(self.content_lines)
        
        while i < n:
            current_line = self.content_lines[i].strip()
            
            # 跳过空行
            if not current_line:
                self.fixed_content.append("\n")
                i += 1
                continue

            # --- 类型 A: 处理带 # 的断行标题 ---
            if current_line.startswith('##'):
                # 尝试前瞻：看下一行非空行是否能补全它
                next_idx = i + 1
                while next_idx < n and not self.content_lines[next_idx].strip():
                    next_idx += 1
                
                if next_idx < n:
                    potential_full_title = current_line + self.content_lines[next_idx].strip()
                    # 检查合并后是否在目录中，或者合并后更像一个标题
                    match_found, _ = self.is_similar(potential_full_title.replace('#', '').strip())
                    if match_found:
                        self.fixed_content.append(potential_full_title.replace('##', '## ') + "\n")
                        i = next_idx + 1
                        continue

            # --- 类型 B: 处理缺失 # 的孤立行 ---
            # 判据：字数短、不在段落中、与目录匹配
            if not current_line.startswith('#') and len(current_line) < 40:
                match_found, std_title = self.is_similar(current_line)
                if match_found:
                    # 补齐 ## 标识符（根据目录层级可调整）
                    self.fixed_content.append(f"## {current_line}\n")
                    i += 1
                    continue

            # 原样保留
            self.fixed_content.append(self.content_lines[i])
            i += 1

    def save_result(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(self.fixed_content)
        print(f"清洗完成，已保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    fixer = MarkdownTitleFixer("F:/SatelliteCorpus/md_books/test.md")
    fixer.load_data()
    fixer.extract_toc()
    fixer.fix_titles()
    fixer.save_result("F:/SatelliteCorpus/md_books/output_fixed.md")