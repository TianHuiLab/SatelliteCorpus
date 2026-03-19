import json
from collections import Counter

samples = [json.loads(l) for l in open('output_v22_sft5/20260305_005946/sft_train.sharegpt.jsonl')]
print(f'=== SFT样本统计 (v2.2最终版) ===')
print(f'总样本数: {len(samples)}')

types = Counter(s.get('type','unknown') for s in samples)
print(f'\n类型分布:')
for t, cnt in types.most_common():
    pct = cnt / len(samples) * 100
    print(f'  {t}: {cnt} ({pct:.1f}%)')

# 三次迭代对比
print(f'\n=== 三次迭代对比 ===')
print(f'v2.0原版(39条): text_summary=23(59%), knowledge_qa=14(36%), concept_explanation=2(5%)')
print(f'v2.2首次(58条): knowledge_qa=30(52%), text_summary=14(24%), concept_explanation=10(17%), causal_reasoning=3(5%), cot_reasoning=1(2%)')
parts = []
for t, cnt in types.most_common():
    parts.append(f'{t}={cnt}({cnt*100//len(samples)}%)')
print(f'v2.2最终({len(samples)}条): ' + ', '.join(parts))

# 展示每种类型的样本示例
print(f'\n=== 各类型样本示例 ===')
shown = set()
for s in samples:
    t = s.get('type','')
    if t not in shown:
        shown.add(t)
        convs = s.get('conversations', [])
        q = convs[0]['value'][:100] if convs else 'N/A'
        a = convs[1]['value'][:120] if len(convs) > 1 else 'N/A'
        print(f'\n--- {t} ---')
        print(f'Q: {q}')
        print(f'A: {a}')

# 检查是否还有未知类型
unknown = [s.get('type','') for s in samples if s.get('type','') not in 
           {'knowledge_qa','concept_explanation','text_summary','causal_reasoning','cot_reasoning','process_qa'}]
if unknown:
    print(f'\n!!! 发现非标准类型: {Counter(unknown)}')
else:
    print(f'\n所有样本类型均为标准类型 ✓')
