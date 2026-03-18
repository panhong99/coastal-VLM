# merge_data.py 최종 수정
import json
import random
from collections import Counter

with open('/home/hail/pan/VLM-project/finetune_data.json') as f:
    real_data = json.load(f)

with open('/home/hail/pan/VLM-project/finetune_data_synthetic.json') as f:
    synthetic_data = json.load(f)

for d in real_data:
    d['source'] = 'real'

# 실제 데이터도 레벨별 분류
real_red    = [d for d in real_data if 'RED'    in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]
real_yellow = [d for d in real_data if 'YELLOW' in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]
real_green  = [d for d in real_data if 'GREEN'  in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]

# 합성 데이터 레벨별 분류
syn_red    = [d for d in synthetic_data if 'RED'    in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]
syn_yellow = [d for d in synthetic_data if 'YELLOW' in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]
syn_green  = [d for d in synthetic_data if 'GREEN'  in d['output'].split('**경보 레벨:** ')[1].split('\n')[0]]

# 목표: RED 20%, YELLOW 50%, GREEN 30%
# 총 60,000개 목표
target_red    = 12000
target_yellow = 30000
target_green  = 18000

merged = (
    real_red    + random.sample(syn_red,    min(target_red    - len(real_red),    len(syn_red)))    +
    real_yellow + random.sample(syn_yellow, min(target_yellow - len(real_yellow), len(syn_yellow))) +
    random.sample(real_green, min(5000, len(real_green))) +
    random.sample(syn_green,  min(target_green - 5000,   len(syn_green)))
)

random.shuffle(merged)

with open('/home/hail/pan/VLM-project/finetune_data_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

alerts  = [d['output'].split('**경보 레벨:** ')[1].split('\n')[0] for d in merged]
sources = Counter([d['source'] for d in merged])

print(f"총 데이터: {len(merged)}개")
print(f"실제/합성 비율: {sources}")
print("\n경보 레벨 분포:")
for k, v in Counter(alerts).items():
    print(f"  {k}: {v}개 ({v/len(merged)*100:.1f}%)")