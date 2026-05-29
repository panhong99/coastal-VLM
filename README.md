# 해안 경비 AI 시스템 (Coastal Surveillance AI)

> Qwen2.5-VL 기반 멀티프레임 VLM + LoRA 파인튜닝을 활용한 해안 감시 자동화 시스템

---

## 📌 프로젝트 개요

해안 CCTV 영상을 실시간으로 분석하여 위협 상황을 자동 탐지하고, 경보 레벨(GREEN / YELLOW / RED)을 판단하여 담당자에게 텔레그램 알림을 전송하는 AI 시스템입니다.

### 주요 기능
- **멀티프레임 영상 분석** : 8프레임 시퀀스를 VLM에 동시 입력하여 시간적 맥락 반영
- **자동 경보 분류** : GREEN(정상) / YELLOW(경계) / RED(위험) 3단계 판단
- **RAG 기반 보고서 생성** : ChromaDB + KR-SBERT를 활용한 상황별 대응 지침 검색
- **텔레그램 실시간 알림** : YELLOW 이상 경보 시 이미지 + 요약 자동 전송
- **Gradio 웹 UI** : 데모 영상 업로드 및 분석 결과 시각화

---

## 🏗️ 파이프라인 흐름

```
영상 입력
   ↓
프레임 추출 (8프레임)
   ↓
픽셀 차분 이상 탐지 (threshold = 5.0)
   ↓
VLM 1차 분석 (Qwen2.5-VL + LoRA)
   ↓
RAG 유사 사례 검색 (ChromaDB + KR-SBERT)
   ↓
최종 보고서 생성 + 경보 레벨 판단
   ↓
텔레그램 알림 전송 (YELLOW / RED)
```

---

## 🧠 모델 및 학습

| 항목 | 내용 |
|------|------|
| 베이스 모델 | Qwen2.5-VL-7B-Instruct |
| 파인튜닝 방식 | LoRA (r=8, alpha=16, dropout=0.05) |
| 학습 데이터 | 실제 해안 데이터 + 합성 데이터 (6,000 멀티프레임 샘플) |
| 입력 형식 | 8프레임 이미지 시퀀스 + 기상/위치 텍스트 |
| 학습 환경 | RTX 4090, DeepSpeed ZeRO-2, bf16 |
| 학습 시간 | 약 6시간 |

### 평가 결과 (미학습 데이터 30개 기준)

| 경보 레벨 | 정확도 |
|-----------|--------|
| 🔴 RED | 9/10 (90%) |
| 🟡 YELLOW | 10/10 (100%) |
| 🟢 GREEN | 6/10 (60%) |
| **전체** | **25/30 (83.3%)** |

---

## 📁 파일 구조

```
VLM-project/
├── Fast_api.py                      # FastAPI 백엔드 (포트 8000)
├── gradio_ui.py                     # Gradio 프론트엔드 (포트 7860)
├── telegram_alert.py                # 텔레그램 알림 모듈
├── rag_pipeline.py                  # RAG 검색 파이프라인
│
├── train_lora.py                    # LoRA 파인튜닝 학습 스크립트
├── make_multiframe_data.py          # 8프레임 학습 데이터 생성
├── make_finetune_data.py            # 실제 데이터 파인튜닝 데이터 생성
├── make_finetune_data_synthetic.py  # 합성 데이터 생성
├── merge_data.py                    # 데이터 병합
├── balance_data.py                  # 클래스 불균형 보정
│
├── build_db.py                      # RAG용 사례 데이터 구성
├── build_vectordb.py                # ChromaDB 벡터DB 구축
├── verify_vectordb.py               # ChromaDB 검증
│
├── eval_multiframe.py               # 멀티프레임 모델 평가
├── ds_config.json                   # DeepSpeed 설정
├── Dockerfile                       # Docker 이미지 빌드
├── docker-compose.yml               # Docker Compose 설정
├── requirements.txt                 # Python 의존성
│
├── models/                          # 베이스 모델 (Qwen2.5-VL-7B) - 별도 다운로드
├── checkpoints/lora_multiframe/     # 학습된 LoRA 체크포인트 - 별도 다운로드
├── chroma_db/                       # ChromaDB 벡터 저장소 - 별도 다운로드
└── demo_videos/                     # 데모용 영상 (GREEN / YELLOW / RED)
```

---

## 🚀 실행 방법

### 방법 1: 로컬 직접 실행

```bash
# 터미널 1 - API 서버
conda activate coastal_vlm
BASE_MODEL_PATH=/path/to/models/Qwen2.5-VL-7B-Instruct \
LORA_MODEL_PATH=/path/to/checkpoints/lora_multiframe/final \
CHROMA_PATH=/path/to/chroma_db \
python Fast_api.py

# 터미널 2 - Gradio UI
conda activate coastal_vlm
DEMO_VIDEOS_DIR=/path/to/demo_videos \
API_URL=http://localhost:8000/analyze_video \
python gradio_ui.py
```

브라우저에서 `http://localhost:7860` 접속

---

### 방법 2: Docker

**사전 준비**: 모델 / 체크포인트 / 벡터DB 파일을 서버에 준비

```
/your/data/path/
├── models/Qwen2.5-VL-7B-Instruct/
├── checkpoints/lora_multiframe/final/
└── chroma_db/
```

**`.env` 파일 생성**

```env
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BASE_MODEL_PATH=/app/models/Qwen2.5-VL-7B-Instruct
LORA_MODEL_PATH=/app/checkpoints/lora_multiframe/final
CHROMA_PATH=/app/chroma_db
```

**Docker Hub에서 Pull 후 실행**

```bash
docker pull panhong472/coastal-surveillance:latest

# API 서버
docker run -d --name coastal-api --gpus all \
  -p 8000:8000 \
  -v /your/data/path/models:/app/models \
  -v /your/data/path/checkpoints:/app/checkpoints \
  -v /your/data/path/chroma_db:/app/chroma_db \
  -e LORA_MODEL_PATH=/app/checkpoints/lora_multiframe/final \
  --env-file .env \
  panhong472/coastal-surveillance:latest python3.11 Fast_api.py

# Gradio UI
docker run -d --name coastal-ui \
  -p 7860:7860 \
  -e API_URL=http://coastal-api:8000/analyze_video \
  --link coastal-api \
  panhong472/coastal-surveillance:latest python3.11 gradio_ui.py
```

브라우저에서 `http://localhost:7860` 접속

**컨테이너 관리**

```bash
docker ps                           # 실행 중 확인
docker stop coastal-api coastal-ui  # 중지
docker start coastal-api coastal-ui # 재시작
```

---

### 외부 공개 URL (ngrok)

```bash
ngrok http 7860
# 출력된 https://xxxx.ngrok-free.app 주소로 외부 접속 가능
```

---

## 🔔 텔레그램 알림 설정

1. [@BotFather](https://t.me/BotFather) 에서 봇 생성 → `TELEGRAM_TOKEN` 발급
2. 본인 또는 그룹 채팅의 Chat ID 확인 → `TELEGRAM_CHAT_ID` 설정
3. YELLOW / RED 경보 감지 시 자동으로 이미지 + 요약 메시지 전송

---

## ⚙️ 재학습 방법

```bash
# 1. 멀티프레임 학습 데이터 생성
python make_multiframe_data.py

# 2. LoRA 학습 (DeepSpeed)
conda activate coastal_vlm
python -m deepspeed.launcher.runner --num_gpus=1 train_lora.py

# 3. 평가
python eval_multiframe.py
```

---

## 📦 환경 요구사항

- Python 3.11
- CUDA 12.8+
- GPU : RTX 4090 권장 (VRAM 24GB 이상)
- Docker + nvidia-container-toolkit (Docker 실행 시)

주요 라이브러리: `transformers`, `peft`, `deepspeed`, `gradio`, `fastapi`, `chromadb`
