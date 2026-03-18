# data_check.py
import json
from collections import Counter

with open('/home/hail/pan/VLM-project/finetune_data.json') as f:
    data = json.load(f)

# 경보 레벨 분포 확인
alerts = [d['output'].split('**경보 레벨:** ')[1].split('\n')[0] for d in data]
print("경보 레벨 분포:")
for k, v in Counter(alerts).items():
    print(f"  {k}: {v}개 ({v/len(data)*100:.1f}%)")

# 탐지 객체 분포 확인
objects = [d['output'].split('**탐지 객체:** ')[1].split(' ')[0] for d in data]
print("\n탐지 객체 분포:")
for k, v in Counter(objects).items():
    print(f"  {k}: {v}개 ({v/len(data)*100:.1f}%)")