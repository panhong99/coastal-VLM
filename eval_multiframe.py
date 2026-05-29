"""
eval_multiframe.py

멀티프레임(8장) 학습된 LoRA 모델 평가
- 학습에 사용하지 않은 샘플(index 6000 이후)에서 30개 추출
- RED / YELLOW / GREEN 각 10개씩
- 추론 포맷: 학습과 동일하게 8장 이미지 → 텍스트
"""

import json
import torch
import random
from pathlib import Path
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ── 경로 설정 ─────────────────────────────────────────────
BASE_MODEL_PATH = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
LORA_MODEL_PATH = "/home/hail/pan/VLM-project/checkpoints/lora_multiframe/final"
DATA_PATH       = "/home/hail/pan/VLM-project/finetune_data_multiframe.json"
RESULT_PATH     = "/home/hail/pan/VLM-project/eval_multiframe_results.json"

TRAIN_SIZE = 6000   # 학습에 사용한 샘플 수

REAL_IMG_DIRS = [
    "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images",
    "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I2_images",
]
SYN_IMG_DIRS = [
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_DT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_NT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/EO_WI_DT",
    "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT",
]


# ── 이미지 맵 ─────────────────────────────────────────────
def build_image_map(dirs):
    img_map = {}
    for d in dirs:
        for f in Path(d).glob("*.jpg"):
            img_map[f.name] = str(f)
    return img_map

def extract_alert_level(text):
    for level in ["RED", "YELLOW", "GREEN"]:
        if level in text:
            return level
    return "UNKNOWN"


# ── 멀티프레임 추론 ───────────────────────────────────────
def inference(model, processor, image_names, input_text, image_map):
    """8프레임 멀티프레임 추론 (학습 포맷과 동일)"""
    image_content = [
        {
            "type":           "image",
            "image":          image_map[img_name],
            "resized_height": 224,
            "resized_width":  224,
        }
        for img_name in image_names
        if img_name in image_map
    ]

    messages = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": input_text}]
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
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    return processor.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


# ── 메인 ─────────────────────────────────────────────────
def main():
    print("=== 멀티프레임 모델 평가 ===\n")

    # 이미지 맵 구성
    print("[1/4] 이미지 경로 매핑 중...")
    image_map = build_image_map(REAL_IMG_DIRS + SYN_IMG_DIRS)
    print(f"  총 {len(image_map)}개 이미지 매핑 완료")

    # 미학습 데이터 로드 (index 6000 이후)
    print("[2/4] 평가 데이터 준비 중...")
    with open(DATA_PATH, encoding="utf-8") as f:
        all_data = json.load(f)

    unseen = all_data[TRAIN_SIZE:]   # 학습에 사용하지 않은 샘플
    # 모든 이미지가 매핑되는 샘플만
    unseen = [d for d in unseen if all(img in image_map for img in d["images"])]
    print(f"  미학습 유효 샘플: {len(unseen)}개")

    # 레벨별 10개씩 균등 샘플링
    by_level = defaultdict(list)
    for d in unseen:
        level = extract_alert_level(d["output"])
        by_level[level].append(d)

    print(f"  레벨 분포 → RED: {len(by_level['RED'])} / YELLOW: {len(by_level['YELLOW'])} / GREEN: {len(by_level['GREEN'])}")

    N = 10
    samples = (
        random.sample(by_level["RED"],    min(N, len(by_level["RED"])))    +
        random.sample(by_level["YELLOW"], min(N, len(by_level["YELLOW"]))) +
        random.sample(by_level["GREEN"],  min(N, len(by_level["GREEN"])))
    )
    random.shuffle(samples)
    print(f"  평가 샘플: {len(samples)}개 (RED {min(N,len(by_level['RED']))} / YELLOW {min(N,len(by_level['YELLOW']))} / GREEN {min(N,len(by_level['GREEN']))})\n")

    # 모델 로드
    print("[3/4] 모델 로딩 중...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    print("  모델 로드 완료\n")

    # 평가
    print("[4/4] 평가 시작")
    print("=" * 60)

    results = []
    correct = 0

    for i, sample in enumerate(samples):
        gt_level = extract_alert_level(sample["output"])

        pred_text  = inference(model, processor, sample["images"], sample["input"], image_map)
        pred_level = extract_alert_level(pred_text)
        is_correct = (pred_level == gt_level)

        if is_correct:
            correct += 1

        mark = "✅" if is_correct else "❌"
        print(f"[{i+1:2d}/{len(samples)}] {mark} 정답: {gt_level:6s} | 예측: {pred_level:6s} | {sample['images'][-1]}")
        if not is_correct:
            print(f"         └ {pred_text[:120]}...")

        results.append({
            "images":     sample["images"],
            "gt_level":   gt_level,
            "pred_level": pred_level,
            "correct":    is_correct,
        })

    # 결과 출력
    print("\n" + "=" * 60)
    print(f"전체 정확도: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
    for level in ["RED", "YELLOW", "GREEN"]:
        level_samples = [r for r in results if r["gt_level"] == level]
        if level_samples:
            lc = sum(1 for r in level_samples if r["correct"])
            print(f"  {level:6s}: {lc}/{len(level_samples)} ({lc/len(level_samples)*100:.0f}%)")

    # 저장
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 완료: {RESULT_PATH}")


if __name__ == "__main__":
    main()
