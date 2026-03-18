# test_base.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json

# 모델 로드
model_path = "./models/Qwen2.5-VL-7B-Instruct"

print("모델 로딩 중...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)
print(f"로드 완료! VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# 샘플 데이터 로드
json_path = "./dataset/open_data/data/Training/jsons/I1_jsons/I1_S0_C5_0001001.json"
img_path  = "./dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0001001.jpg"

with open(json_path) as f:
    meta = json.load(f)

weather = meta["meta"]["weather"]
caption = meta["annotations"][0]["caption"]
print(f"\n정답 캡션: {caption}")

# 추론
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": f"""해안 감시 카메라 이미지입니다.
기상정보: 기온 {weather[0]}°C, 풍향 {weather[1]}°, 풍속 {weather[2]}m/s, 습도 {weather[6]}%

다음 형식으로 상황을 분석해주세요:
- 탐지 객체: (선박 종류, 수량)
- 기상 위험도: (낮음/보통/높음)
- 경보 레벨: (GREEN/YELLOW/RED)
- 상황 요약:
- 권고 조치:"""}
        ]
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=300)

response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n모델 응답:\n{response}")