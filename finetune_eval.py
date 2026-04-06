# # evaluate.py
# import json
# import torch
# import random
# from pathlib import Path
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from peft import PeftModel
# from qwen_vl_utils import process_vision_info

# # ── 경로 설정 ──────────────────────────────────────────
# BASE_MODEL_PATH  = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
# LORA_MODEL_PATH  = "/home/hail/pan/VLM-project/checkpoints/lora/final"
# DATA_PATH        = "/home/hail/pan/VLM-project/finetune_data_merged.json"

# REAL_IMG_DIRS = [
#     "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images",
#     "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I2_images",
# ]
# SYN_IMG_DIRS = [
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_DT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_NT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_WI_DT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT",
# ]

# def build_image_map(dirs):
#     img_map = {}
#     for d in dirs:
#         for f in Path(d).glob("*.jpg"):
#             img_map[f.name] = str(f)
#     return img_map

# print("이미지 경로 매핑 중...")
# IMAGE_MAP = build_image_map(REAL_IMG_DIRS + SYN_IMG_DIRS)

# # ── 모델 로드 ──────────────────────────────────────────
# print("베이스 모델 로딩 중...")
# processor = AutoProcessor.from_pretrained(
#     BASE_MODEL_PATH,
#     min_pixels=256 * 28 * 28,
#     max_pixels=512 * 28 * 28
# )

# base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     BASE_MODEL_PATH,
#     dtype=torch.bfloat16,
#     device_map="auto"
# )

# print("LoRA 모델 로딩 중...")
# lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
# lora_model.eval()

# # ── 추론 함수 ──────────────────────────────────────────
# def inference(model, img_path, input_text):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type":           "image",
#                     "image":          img_path,
#                     "resized_height": 224,
#                     "resized_width":  224,
#                 },
#                 {"type": "text", "text": input_text}
#             ]
#         }
#     ]

#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, _ = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         return_tensors="pt"
#     ).to("cuda")

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             do_sample=False,      # greedy decoding
#             temperature=1.0,
#         )

#     response = processor.decode(
#         output[0][inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )
#     return response

# # ── 평가 ──────────────────────────────────────────────
# def extract_alert_level(text):
#     for level in ["RED", "YELLOW", "GREEN"]:
#         if level in text:
#             return level
#     return "UNKNOWN"

# print("\n테스트 샘플 로딩 중...")
# with open(DATA_PATH) as f:
#     data = json.load(f)

# # 유효한 샘플만 필터링 후 랜덤 30개 선택
# valid = [d for d in data if d['image'] in IMAGE_MAP]
# samples = random.sample(valid, 30)

# # 경보 레벨별 10개씩 균등 샘플링
# from collections import defaultdict
# by_level = defaultdict(list)
# for d in valid:
#     level = extract_alert_level(d['output'])
#     by_level[level].append(d)

# samples = (
#     random.sample(by_level['RED'],    10) +
#     random.sample(by_level['YELLOW'], 10) +
#     random.sample(by_level['GREEN'],  10)
# )

# print(f"평가 샘플: {len(samples)}개 (RED 10 / YELLOW 10 / GREEN 10)")
# print("\n" + "="*60)

# # ── 결과 수집 ──────────────────────────────────────────
# results = []
# correct = 0

# for i, sample in enumerate(samples):
#     img_path   = IMAGE_MAP[sample['image']]
#     gt_level   = extract_alert_level(sample['output'])

#     print(f"\n[{i+1}/30] {sample['image']} | 정답: {gt_level}")

#     # LoRA 모델 추론
#     pred = inference(lora_model, img_path, sample['input'])
#     pred_level = extract_alert_level(pred)

#     is_correct = (pred_level == gt_level)
#     if is_correct:
#         correct += 1

#     print(f"예측: {pred_level} | {'✅' if is_correct else '❌'}")
#     print(f"출력:\n{pred[:200]}...")

#     results.append({
#         "image":      sample['image'],
#         "gt_level":   gt_level,
#         "pred_level": pred_level,
#         "correct":    is_correct,
#         "output":     pred
#     })

# # ── 최종 결과 ──────────────────────────────────────────
# print("\n" + "="*60)
# print(f"전체 정확도: {correct}/30 ({correct/30*100:.1f}%)")

# # 레벨별 정확도
# for level in ["RED", "YELLOW", "GREEN"]:
#     level_samples = [r for r in results if r['gt_level'] == level]
#     level_correct = sum(1 for r in level_samples if r['correct'])
#     print(f"{level} 정확도: {level_correct}/10 ({level_correct*10:.0f}%)")

# # 결과 저장
# with open('/home/hail/pan/VLM-project/eval_results.json', 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# print("\n결과 저장 완료: eval_results.json")



# # evaluate_unseen.py
# import json
# import torch
# import random
# from pathlib import Path
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from peft import PeftModel
# from qwen_vl_utils import process_vision_info
# from collections import defaultdict

# BASE_MODEL_PATH = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
# LORA_MODEL_PATH = "/home/hail/pan/VLM-project/checkpoints/lora/final"

# REAL_IMG_DIRS = [
#     "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images",
#     "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I2_images",
# ]
# SYN_IMG_DIRS = [
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_DT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_NT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_WI_DT",
#     "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT",
# ]

# def build_image_map(dirs):
#     img_map = {}
#     for d in dirs:
#         for f in Path(d).glob("*.jpg"):
#             img_map[f.name] = str(f)
#     return img_map

# print("이미지 경로 매핑 중...")
# IMAGE_MAP = build_image_map(REAL_IMG_DIRS + SYN_IMG_DIRS)

# # 미학습 데이터 추출
# with open('/home/hail/pan/VLM-project/finetune_data_merged.json') as f:
#     train_data = json.load(f)
# train_images = set(d['image'] for d in train_data)

# with open('/home/hail/pan/VLM-project/finetune_data.json') as f:
#     all_data = json.load(f)

# unseen = [d for d in all_data
#           if d['image'] not in train_images
#           and d['image'] in IMAGE_MAP]

# print(f"미학습 유효 데이터: {len(unseen)}개")

# # 경보 레벨별 균등 샘플링
# def extract_alert_level(text):
#     for level in ["RED", "YELLOW", "GREEN"]:
#         if level in text:
#             return level
#     return "UNKNOWN"

# # 합성 데이터에서 미학습 데이터 추가 추출
# with open('/home/hail/pan/VLM-project/finetune_data_synthetic.json') as f:
#     syn_data = json.load(f)

# syn_unseen = [d for d in syn_data
#               if d['image'] not in train_images
#               and d['image'] in IMAGE_MAP]

# by_level_syn = defaultdict(list)
# for d in syn_unseen:
#     level = extract_alert_level(d['output'])
#     by_level_syn[level].append(d)

# print(f"합성 미학습 - RED: {len(by_level_syn['RED'])} / YELLOW: {len(by_level_syn['YELLOW'])} / GREEN: {len(by_level_syn['GREEN'])}")

# by_level = defaultdict(list)
# for d in unseen:
#     level = extract_alert_level(d['output'])
#     by_level[level].append(d)

# print(f"레벨 분포 - RED: {len(by_level['RED'])} / YELLOW: {len(by_level['YELLOW'])} / GREEN: {len(by_level['GREEN'])}")

# # 각 레벨에서 10개씩 (없으면 있는 만큼)
# n = 10
# samples = (
#     random.sample(by_level_syn['RED'],    min(n, len(by_level_syn['RED'])))    +
#     random.sample(by_level_syn['YELLOW'], min(n, len(by_level_syn['YELLOW']))) +
#     random.sample(by_level['GREEN'],      min(n, len(by_level['GREEN'])))
# )
# random.shuffle(samples)
# print(f"평가 샘플: {len(samples)}개\n")

# # 모델 로드
# print("모델 로딩 중...")
# processor = AutoProcessor.from_pretrained(
#     BASE_MODEL_PATH,
#     min_pixels=256 * 28 * 28,
#     max_pixels=512 * 28 * 28
# )
# base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     BASE_MODEL_PATH,
#     dtype=torch.bfloat16,
#     device_map="auto"
# )
# lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
# lora_model.eval()

# def inference(model, img_path, input_text):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type":           "image",
#                     "image":          img_path,
#                     "resized_height": 224,
#                     "resized_width":  224,
#                 },
#                 {"type": "text", "text": input_text}
#             ]
#         }
#     ]
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, _ = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         return_tensors="pt"
#     ).to("cuda")

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             do_sample=False,
#         )
#     return processor.decode(
#         output[0][inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )

# # 평가
# print("="*60)
# results  = []
# correct  = 0

# for i, sample in enumerate(samples):
#     img_path = IMAGE_MAP[sample['image']]
#     gt_level = extract_alert_level(sample['output'])

#     pred       = inference(lora_model, img_path, sample['input'])
#     pred_level = extract_alert_level(pred)
#     is_correct = (pred_level == gt_level)

#     if is_correct:
#         correct += 1

#     print(f"[{i+1}/{len(samples)}] {sample['image']}")
#     print(f"  정답: {gt_level} | 예측: {pred_level} | {'✅' if is_correct else '❌'}")
#     if not is_correct:
#         print(f"  예측 출력: {pred[:150]}...")

#     results.append({
#         "image":      sample['image'],
#         "gt_level":   gt_level,
#         "pred_level": pred_level,
#         "correct":    is_correct,
#     })

# # 최종 결과
# print("\n" + "="*60)
# print(f"전체 정확도: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
# for level in ["RED", "YELLOW", "GREEN"]:
#     level_samples = [r for r in results if r['gt_level'] == level]
#     if level_samples:
#         level_correct = sum(1 for r in level_samples if r['correct'])
#         print(f"{level} 정확도: {level_correct}/{len(level_samples)} ({level_correct/len(level_samples)*100:.0f}%)")

# # 저장
# with open('/home/hail/pan/VLM-project/eval_unseen_results.json', 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
# print("\n결과 저장 완료: eval_unseen_results.json")

# edge_case_test.py
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import random

BASE_MODEL_PATH = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
LORA_MODEL_PATH = "/home/hail/pan/VLM-project/checkpoints/lora/final"

processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)
lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
lora_model.eval()

def inference(img_path, input_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type":           "image",
                    "image":          img_path,
                    "resized_height": 224,
                    "resized_width":  224,
                },
                {"type": "text", "text": input_text}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output = lora_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    return processor.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

# 실제 이미지 랜덤 1장 가져오기
img_dir = "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images"
img_path = str(random.choice(list(Path(img_dir).glob("*.jpg"))))

# 경계선 케이스 테스트
edge_cases = [
    {
        "name": "야간 + 군함 + 기상 양호 (YELLOW 예상)",
        "input": "해안 감시 카메라 이미지입니다. [연평도 / 야간]\n기상정보: 기온 15.0°C, 풍향 270°, 풍속 4.0m/s, 강수량 0mm, 습도 60%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
    },
    {
        "name": "주간 + 어선 + 강풍 (YELLOW 예상)",
        "input": "해안 감시 카메라 이미지입니다. [백령도 / 주간]\n기상정보: 기온 10.0°C, 풍향 180°, 풍속 13.0m/s, 강수량 5mm, 습도 85%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
    },
    {
        "name": "야간 + 미확인 선박 + 폭풍 (RED 예상)",
        "input": "해안 감시 카메라 이미지입니다. [연평도 / 야간]\n기상정보: 기온 5.0°C, 풍향 315°, 풍속 20.0m/s, 강수량 30mm, 습도 95%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
    },
    {
        "name": "주간 + 어선 + 기상 양호 (GREEN 예상)",
        "input": "해안 감시 카메라 이미지입니다. [연평도 / 주간]\n기상정보: 기온 25.0°C, 풍향 90°, 풍속 2.0m/s, 강수량 0mm, 습도 55%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
    },
]

print("="*60)
print("경계선 케이스 테스트")
print(f"사용 이미지: {img_path}")
print("="*60)

for case in edge_cases:
    print(f"\n[케이스] {case['name']}")
    result = inference(img_path, case['input'])
    print(f"출력:\n{result}")
    print("-"*40)