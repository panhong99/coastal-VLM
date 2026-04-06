# rag_pipeline.py
import torch
import json
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── 경로 설정 ──────────────────────────────────────────
BASE_MODEL_PATH = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
LORA_MODEL_PATH = "/home/hail/pan/VLM-project/checkpoints/lora/final"
CHROMA_PATH     = "/home/hail/pan/VLM-project/chroma_db"

# ── 모델 로드 ──────────────────────────────────────────
print("VLM 모델 로딩 중...")
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
vlm_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
vlm_model.eval()
print("VLM 로드 완료!")

# ── 벡터DB 로드 ────────────────────────────────────────
print("ChromaDB 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={"device": "cuda"}
)
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)
print(f"ChromaDB 로드 완료! ({vectordb._collection.count()}개 벡터)")

# ── VLM 1차 분석 ───────────────────────────────────────
def vlm_analyze(img_path, weather_input):
    """VLM으로 이미지 1차 분석"""
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
                {"type": "text", "text": weather_input}
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
        output = vlm_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    return processor.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

# ── RAG 검색 ───────────────────────────────────────────
def rag_search(vlm_output, k=3):
    """VLM 분석 결과로 유사 사례 검색"""

    # 경보 레벨 추출
    alert_level = "GREEN"
    for level in ["RED", "YELLOW", "GREEN"]:
        if level in vlm_output:
            alert_level = level
            break

    # 탐지 객체 추출
    object_type = None
    for obj in ["군함", "어선", "상선", "드론", "오물폭탄", "삐라", "항공기"]:
        if obj in vlm_output:
            object_type = obj
            break

    # 필터 구성
    if object_type:
        filter_dict = {"object": {"$eq": object_type}}
    else:
        filter_dict = None

    # 유사 사례 검색
    results = vectordb.similarity_search_with_relevance_scores(
        vlm_output,
        k=k,
        filter=filter_dict
    )
    return results, alert_level, object_type

# ── 최종 리포트 생성 ───────────────────────────────────
def generate_final_report(img_path, weather_input):
    """전체 RAG 파이프라인 실행"""

    print("\n" + "="*60)
    print("1단계: VLM 이미지 분석 중...")
    vlm_output = vlm_analyze(img_path, weather_input)
    print(f"VLM 분석 결과:\n{vlm_output}")

    print("\n2단계: 유사 사례 검색 중...")
    similar_cases, alert_level, object_type = rag_search(vlm_output)

    print(f"\n유사 사례 TOP {len(similar_cases)}개:")
    case_texts = []
    for i, (doc, score) in enumerate(similar_cases):
        print(f"  [{i+1}] 유사도: {score:.3f} | "
              f"경보: {doc.metadata['alert_level']} | "
              f"객체: {doc.metadata['object']} | "
              f"시간: {doc.metadata['time']}")
        case_texts.append(doc.page_content)

    # 유사 사례 기반 경보 레벨 재검토
    case_levels  = [doc.metadata['alert_level'] for doc, _ in similar_cases]
    level_counts = {
        "RED":    case_levels.count("RED"),
        "YELLOW": case_levels.count("YELLOW"),
        "GREEN":  case_levels.count("GREEN")
    }

    # 과반수 이상이면 해당 레벨로 조정
    final_level = alert_level
    if level_counts["RED"] >= 2:
        final_level = "RED"
    elif level_counts["YELLOW"] >= 2 and alert_level == "GREEN":
        final_level = "YELLOW"

    print(f"\n3단계: 최종 리포트 생성...")
    print(f"  VLM 판단: {alert_level}")
    print(f"  사례 분포: {level_counts}")
    print(f"  최종 판단: {final_level}")

    # 최종 리포트
    report = {
        "vlm_analysis":   vlm_output,
        "similar_cases":  len(similar_cases),
        "case_levels":    level_counts,
        "vlm_level":      alert_level,
        "final_level":    final_level,
        "level_changed":  alert_level != final_level,
        "report":         vlm_output.replace(
            f"**경보 레벨:** {alert_level}",
            f"**경보 레벨:** {final_level} (RAG 보정)"
        ) if alert_level != final_level else vlm_output
    }

    print("\n" + "="*60)
    print("최종 리포트:")
    print(report['report'])
    if report['level_changed']:
        print(f"\n⚠️  RAG 보정: {alert_level} → {final_level}")
        print(f"   근거: 유사 사례 {level_counts[final_level]}건이 {final_level} 판정")

    return report

# ── 테스트 실행 ────────────────────────────────────────
if __name__ == "__main__":
    # 테스트 이미지
    test_images = [
        {
            "img": "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0120002.jpg",
            "weather": "해안 감시 카메라 이미지입니다. [연평도 / 야간]\n기상정보: 기온 15.0°C, 풍향 270°, 풍속 15.0m/s, 강수량 5mm, 습도 80%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요.",
            "desc": "야간 + 강풍"
        },
        {
            "img": "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0001001.jpg",
            "weather": "해안 감시 카메라 이미지입니다. [연평도 / 주간]\n기상정보: 기온 28.0°C, 풍향 180°, 풍속 2.0m/s, 강수량 0mm, 습도 60%\n\n다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요.",
            "desc": "주간 + 기상 양호"
        },
    ]

    for test in test_images:
        print(f"\n{'#'*60}")
        print(f"테스트: {test['desc']}")
        report = generate_final_report(test['img'], test['weather'])
        print(f"{'#'*60}")