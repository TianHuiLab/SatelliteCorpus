#!/usr/bin/env python3
"""检查SFT样本质量"""
import json, sys, glob, os
from collections import Counter

# 找到最新的输出目录
output_dirs = sorted(glob.glob("output/*/sft_train.sharegpt.jsonl"))
if not output_dirs:
    print("未找到SFT输出文件")
    sys.exit(1)

path = output_dirs[-1]
print(f"检查文件: {path}\n")

with open(path, 'r') as f:
    samples = [json.loads(l) for l in f]

print(f"总样本数: {len(samples)}\n")

# 类型分布
types = Counter(s['type'] for s in samples)
print("类型分布:")
for t, c in types.items():
    print(f"  {t}: {c}")
print()

# 诊断分布
diag = Counter(s.get('diagnosis', '') for s in samples)
print("诊断分布:")
for d, c in diag.items():
    print(f"  {d}: {c}")
print()

# 展示每种类型的一个样本
shown = set()
for s in samples:
    if s['type'] not in shown:
        shown.add(s['type'])
        print(f"{'='*60}")
        print(f"类型: {s['type']}")
        print(f"诊断: {s.get('diagnosis', '')}")
        print(f"知识密度: {s.get('metadata', {}).get('knowledge_density', '')}")
        print(f"匹配类别: {s.get('metadata', {}).get('matched_categories', [])}")
        print(f"Q字符数: {s.get('metadata', {}).get('char_count_q', 0)}")
        print(f"A字符数: {s.get('metadata', {}).get('char_count_a', 0)}")
        print(f"\nQ: {s['conversations'][0]['value']}")
        print(f"\nA: {s['conversations'][1]['value'][:300]}...")
        print()

# 查看过滤详情
filter_path = os.path.join(os.path.dirname(path), "filter_details.jsonl")
if os.path.exists(filter_path):
    print(f"\n{'='*60}")
    print("知识密度过滤详情:")
    with open(filter_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            status = "通过" if d.get('should_generate') else "过滤"
            print(f"  [{status}] {d.get('chunk_id', '')[:60]} | 密度={d.get('knowledge_density')} | 原因={d.get('reason', '')[:50]}")

# 查看被过滤的chunk
filtered_path = os.path.join(os.path.dirname(path), "filtered_chunks.jsonl")
if os.path.exists(filtered_path):
    print(f"\n被过滤的chunk:")
    with open(filtered_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            print(f"  {d.get('section_title', '')} ({d.get('char_count', 0)}字符): {d.get('content', '')[:80]}...")
