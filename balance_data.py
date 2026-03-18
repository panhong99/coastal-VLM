# balance_data.py
import json
import random
from collections import defaultdict

with open('/home/hail/pan/VLM-project/finetune_data.json') as f:
    data = json.load(f)

# 객체별로 분류
by_object = defaultdict(list)
for d in data:
    obj = d['output'].split('**탐지 객체:** ')[1].split(' ')[0]
    by_object[obj].append(d)

# 언더샘플링 (각 클래스 최대 3000개)
MAX_PER_CLASS = 3000
balanced = []
for obj, samples in by_object.items():
    if len(samples) > MAX_PER_CLASS:
        samples = random.sample(samples, MAX_PER_CLASS)
    balanced.extend(samples)

# RED 케이스 합성 (군함 데이터 기반으로 악천후 조건 추가)
red_cases = []
gunham_data = by_object["군함"][:500]  # 군함 500개 가져와서

for d in gunham_data:
    red = d.copy()
    # output에서 경보레벨/기상/권고조치 교체
    red['output'] = red['output'].replace(
        '**기상 상태:**', '**기상 상태:** ⚠️'
    )
    red['output'] = red['output'].replace(
        '위험도 낮음', '위험도 높음'
    ).replace(
        '위험도 보통', '위험도 높음'
    )
    red['output'] = red['output'].replace(
        '**경보 레벨:** YELLOW', '**경보 레벨:** RED'
    ).replace(
        '**경보 레벨:** GREEN', '**경보 레벨:** RED'
    )
    red['output'] = red['output'].replace(
        '**권고 조치:** 경계 강화. 군함 식별 및 이동 경로 추적 요망.',
        '**권고 조치:** 즉각 상황 보고. 군함에 대한 긴급 대응 절차 가동.'
    )
    # input에 악천후 조건 반영
    red['input'] = red['input'].replace(
        '풍속 2.', '풍속 15.'
    ).replace(
        '풍속 1.', '풍속 15.'
    ).replace(
        '풍속 3.', '풍속 15.'
    )
    red_cases.append(red)

balanced.extend(red_cases)
random.shuffle(balanced)

# 저장
output_path = '/home/hail/pan/VLM-project/finetune_data_balanced.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

# 결과 확인
alerts = [d['output'].split('**경보 레벨:** ')[1].split('\n')[0] for d in balanced]
objects = [d['output'].split('**탐지 객체:** ')[1].split(' ')[0] for d in balanced]

from collections import Counter
print(f"총 데이터: {len(balanced)}개")
print("\n경보 레벨 분포:")
for k, v in Counter(alerts).items():
    print(f"  {k}: {v}개 ({v/len(balanced)*100:.1f}%)")
print("\n탐지 객체 분포:")
for k, v in Counter(objects).items():
    print(f"  {k}: {v}개 ({v/len(balanced)*100:.1f}%)")