# Coastal Surveillance AI System

해안 CCTV 영상을 실시간으로 분석하여 위협을 탐지하고 경보를 발령하는 AI 파이프라인.

**모델:** Qwen2.5-VL-7B-Instruct + LoRA 파인튜닝  
**경보 레벨:** GREEN (정상) / YELLOW (주의) / RED (위험)

---

## 파이프라인 흐름

```
영상 입력 → 프레임 추출 → 이상 탐지(픽셀 차분) → VLM 1차 분석
→ RAG 유사 사례 검색 → VLM 최종 분석 → 리포트 생성 → 텔레그램 경보
```

---

## 파일 설명

### 서비스

| 파일 | 설명 |
|---|---|
| `Fast_api.py` | FastAPI 백엔드. 영상 수신 → 전체 파이프라인 실행 → JSON 반환 |
| `gradio_ui.py` | Gradio 데모 UI. 영상 업로드, 기상 정보 입력, 결과 시각화 |
| `telegram_alert.py` | YELLOW/RED 경보 시 텔레그램 봇으로 이미지+요약 자동 전송 |
| `rag_pipeline.py` | ChromaDB 기반 RAG. 유사 과거 사례 검색 및 경보 레벨 보정 |

### 학습 데이터 생성

| 파일 | 설명 |
|---|---|
| `make_finetune_data.py` | 실제 데이터(open_data) 기반 파인튜닝 데이터 생성 |
| `make_finetune_data_synthetic.py` | 합성 데이터 기반 파인튜닝 데이터 생성 |
| `merge_data.py` | 실제 + 합성 데이터를 병합하여 `finetune_data_merged.json` 생성 |
| `balance_data.py` | GREEN/YELLOW/RED 클래스 불균형 보정 |
| `data_check.py` | 데이터 분포 및 형식 검증 |

### 학습

| 파일 | 설명 |
|---|---|
| `train_lora.py` | LoRA 파인튜닝 메인 스크립트 (DeepSpeed ZeRO-2 사용) |
| `ds_config.json` | DeepSpeed 설정 (ZeRO-2, bf16, gradient checkpointing) |
| `finetune_eval.py` | 파인튜닝 모델 평가 스크립트 |

### 벡터 DB

| 파일 | 설명 |
|---|---|
| `build_db.py` | RAG용 사례 데이터 구성 |
| `build_vectordb.py` | ChromaDB에 임베딩 저장 (KR-SBERT 사용) |
| `verify_vectordb.py` | ChromaDB 저장 내용 검증 |

### 기타

| 파일 | 설명 |
|---|---|
| `make_demo_video.py` | 시퀀스 이미지로 데모 영상 생성 |
| `realtime_demo.py` | 실시간 웹캠 데모 (개발용) |
| `test.py` / `test_telegram.py` | API 및 텔레그램 동작 확인용 테스트 |

---

## 학습 데이터

- **총 샘플:** 60,000개
- **분포:** GREEN 18,000 / YELLOW 30,000 / RED 12,000
- **입력 형식:** 시퀀스 이미지를 멀티프레임으로 묶어 short video 형태로 구성
- **프롬프트 형식:**
  ```
  해안 감시 카메라 이미지입니다. [위치 / 시간대]
  기상정보: 기온 X°C, 풍향 Y°, 풍속 Zm/s, 강수량 Wmm, 습도 V%
  
  다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요.
  ```
- **출력 형식:** 탐지 객체 / 촬영 환경 / 기상 상태 / 경보 레벨 / 상황 요약 / 권고 조치

---

## LoRA 하이퍼파라미터

| 항목 | 값 |
|---|---|
| base model | Qwen2.5-VL-7B-Instruct |
| lora_r | 8 |
| lora_alpha | 16 |
| lora_dropout | 0.05 |
| learning_rate | 2e-4 |
| epochs | 3 |
| batch size | 1 (gradient accumulation 8) |
| precision | bf16 |
| optimizer offload | DeepSpeed ZeRO-2 (CPU offload) |

체크포인트: `checkpoints/lora/` (6750 / 13500 / 20250 step, final)

---

## 디렉토리 구조

```
VLM-project/
├── Fast_api.py              # 백엔드 API
├── gradio_ui.py             # 데모 UI
├── telegram_alert.py        # 텔레그램 경보
├── rag_pipeline.py          # RAG 파이프라인
├── train_lora.py            # LoRA 학습
├── ds_config.json           # DeepSpeed 설정
├── finetune_data_merged.json # 학습 데이터 (실제+합성)
├── checkpoints/lora/        # LoRA 체크포인트
├── chroma_db/               # ChromaDB 벡터 저장소
├── models/                  # 베이스 모델 (Qwen2.5-VL-7B)
├── dataset/
│   ├── open_data/           # 실제 해안 감시 이미지
│   └── synthetic_data/      # 합성 데이터
└── demo_videos/             # 데모용 영상 (GREEN/YELLOW/RED 각 3개)
```

---

## 실행 방법

```bash
# 환경변수 설정
export TELEGRAM_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# FastAPI 서버 실행 (포트 8000)
python Fast_api.py

# Gradio UI 실행 (포트 7860)
python gradio_ui.py
```
