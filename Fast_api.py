# Fast_api.py
import torch
import json
import cv2
import numpy as np
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from telegram_alert import send_alert as telegram_send_alert
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Path configuration ─────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/home/hail/pan/VLM-project/models/Qwen2.5-VL-7B-Instruct"
LORA_MODEL_PATH = "/home/hail/pan/VLM-project/checkpoints/lora/final"
CHROMA_PATH     = "/home/hail/pan/VLM-project/chroma_db"

# Anomaly detection threshold (0~255 scale, mean pixel difference between frames)
# Real CCTV footage has subtle motion (waves, small objects) — keep this low.
# scene0008 warship video scores ~8.86, so threshold must be below that.
ANOMALY_THRESHOLD = 5.0

app = FastAPI(
    title="Coastal Surveillance AI API",
    description="VLM + RAG based coastal CCTV situation analysis system",
    version="2.0.0"
)

# ── Global models (loaded once at startup) ─────────────────────────────────────
vlm_model  = None
processor  = None
vectordb   = None
embeddings = None


@app.on_event("startup")
async def load_models():
    global vlm_model, processor, vectordb, embeddings

    print("Loading VLM model...")
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
    print("VLM loaded!")

    print("Loading ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        model_kwargs={"device": "cuda"}
    )
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print(f"ChromaDB loaded! ({vectordb._collection.count()} vectors)")


# ── Step 1: Frame extraction ───────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = 8) -> list[str]:
    """
    Extract evenly-spaced frames from a video file.
    Returns a list of temporary image file paths.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    # Pick evenly-spaced indices across the video
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)

    frame_paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp.name, frame)
        frame_paths.append(tmp.name)

    cap.release()
    return frame_paths


# ── Step 2: Anomaly detection (lightweight, no extra model needed) ─────────────
def detect_anomaly(frame_paths: list[str]) -> tuple[bool, float]:
    """
    Detect anomaly by computing mean absolute pixel difference between frames.
    Returns (anomaly_detected, score).

    Logic:
      - Convert each frame to grayscale
      - Compute mean |diff| between consecutive frame pairs
      - If max diff score exceeds ANOMALY_THRESHOLD → anomaly detected
    """
    if len(frame_paths) < 2:
        # Cannot compare with only one frame; assume anomaly to be safe
        return True, 0.0

    frames_gray = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames_gray.append(gray)

    if len(frames_gray) < 2:
        return True, 0.0

    diff_scores = []
    for i in range(len(frames_gray) - 1):
        diff = np.abs(frames_gray[i + 1] - frames_gray[i])
        diff_scores.append(diff.mean())

    max_score = float(np.max(diff_scores))
    anomaly   = max_score > ANOMALY_THRESHOLD
    return anomaly, max_score


# ── Step 3: VLM analysis (accepts multiple frames as video) ────────────────────
def vlm_analyze(frame_paths: list[str], input_text: str) -> str:
    """
    Run VLM inference on multiple frames.
    Qwen2.5-VL accepts a list of images as a video sequence.
    """
    # Build video content using multiple frames
    video_content = [
        {
            "type":           "image",
            "image":          fp,
            "resized_height": 224,
            "resized_width":  224,
        }
        for fp in frame_paths
    ]
    video_content.append({"type": "text", "text": input_text})

    messages = [{"role": "user", "content": video_content}]

    text         = processor.apply_chat_template(
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


def extract_alert_level(text: str) -> str:
    for level in ["RED", "YELLOW", "GREEN"]:
        if level in text:
            return level
    return "UNKNOWN"


# ── Helper: pick the most visually changed frame to attach to alert ───────────
def _pick_alert_frame(frame_paths: list[str]) -> str:
    """Return the frame with the largest pixel difference from its neighbour."""
    if len(frame_paths) == 1:
        return frame_paths[0]

    best_path  = frame_paths[0]
    best_score = 0.0
    prev_gray  = None

    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            score = float(np.abs(gray - prev_gray).mean())
            if score > best_score:
                best_score = score
                best_path  = fp
        prev_gray = gray

    return best_path


# ── Step 4: RAG case search ────────────────────────────────────────────────────
def rag_search(query: str, object_type: str = None, k: int = 3):
    filter_dict = {"object": {"$eq": object_type}} if object_type else None
    return vectordb.similarity_search(query, k=k, filter=filter_dict)


# ── API endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Coastal Surveillance AI API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status":   "healthy",
        "vlm":      vlm_model is not None,
        "vectordb": vectordb is not None,
        "vectors":  vectordb._collection.count() if vectordb else 0
    }


@app.post("/analyze_video")
async def analyze_video(
    video:       UploadFile = File(...),
    location:    str        = Form(default="연평도"),
    time_of_day: str        = Form(default="주간"),
    temperature: float      = Form(default=20.0),
    wind_dir:    float      = Form(default=180.0),
    wind_speed:  float      = Form(default=5.0),
    rainfall:    float      = Form(default=0.0),
    humidity:    float      = Form(default=60.0),
):
    # Save uploaded video to a temp file
    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        video_path = tmp.name

    frame_paths = []
    try:
        # ── Step 1: Extract frames ─────────────────────────────────────────────
        frame_paths = extract_frames(video_path, max_frames=8)
        if not frame_paths:
            return JSONResponse({"status": "error", "message": "Failed to extract frames"}, status_code=400)

        # ── Step 2: Anomaly detection ──────────────────────────────────────────
        anomaly_detected, anomaly_score = detect_anomaly(frame_paths)

        if not anomaly_detected:
            return JSONResponse({
                "status":          "no_anomaly",
                "message":         "No anomaly detected — continuing surveillance",
                "anomaly_score":   round(anomaly_score, 2),
                "frames_analyzed": len(frame_paths),
            })

        # ── Step 3: VLM first analysis ─────────────────────────────────────────
        input_text = (
            f"해안 감시 카메라 영상입니다. [{location} / {time_of_day}]\n"
            f"기상정보: 기온 {temperature}°C, 풍향 {wind_dir}°, "
            f"풍속 {wind_speed}m/s, 강수량 {rainfall}mm, 습도 {humidity}%\n\n"
            f"다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
        )

        first_analysis = vlm_analyze(frame_paths, input_text)
        first_level    = extract_alert_level(first_analysis)

        # Extract detected object type for RAG filtering
        object_type = None
        for obj in ["군함", "어선", "상선", "드론", "오물폭탄", "삐라", "항공기"]:
            if obj in first_analysis:
                object_type = obj
                break

        # ── Check threat level before proceeding ──────────────────────────────
        # Training data contains YELLOW (warship, threat) and GREEN (normal) only.
        # RED was never in training data so the model will never output it.
        # YELLOW is the highest threat level — trigger RAG + report on YELLOW or RED.
        # GREEN means a normal/non-threatening object — skip the alarm.
        if first_level not in ("RED", "YELLOW"):
            return JSONResponse({
                "status":          "no_threat",
                "message":         "Object detected but no threat — continuing surveillance",
                "first_level":     first_level,
                "first_analysis":  first_analysis,
                "object_type":     object_type,
                "anomaly_score":   round(anomaly_score, 2),
                "frames_analyzed": len(frame_paths),
            })

        # ── Step 4: RAG case search ────────────────────────────────────────────
        similar_cases = rag_search(first_analysis, object_type)
        case_texts    = "\n\n".join([
            f"[Case {i+1}] Alert:{doc.metadata['alert_level']} | "
            f"Object:{doc.metadata['object']} | "
            f"Time:{doc.metadata['time']}\n{doc.page_content[:200]}"
            for i, doc in enumerate(similar_cases)
        ])

        # ── Step 5: VLM final analysis (with RAG context) ─────────────────────
        final_prompt = (
            f"해안 감시 카메라 영상입니다. [{location} / {time_of_day}]\n"
            f"기상정보: 기온 {temperature}°C, 풍향 {wind_dir}°, "
            f"풍속 {wind_speed}m/s, 강수량 {rainfall}mm, 습도 {humidity}%\n\n"
            f"[1차 분석 결과]\n{first_analysis}\n\n"
            f"[유사 과거 사례]\n{case_texts}\n\n"
            f"1차 분석 결과와 유사 과거 사례를 참고하여 최종 대응 리포트를 작성하세요."
        )

        final_report = vlm_analyze(frame_paths, final_prompt)
        final_level  = extract_alert_level(final_report)

        # ── Step 5: Send Telegram alert with the most anomalous frame ──────────
        # Pick the frame with the highest difference score as the emergency image
        alert_frame = _pick_alert_frame(frame_paths)
        try:
            telegram_send_alert(
                alert_level = final_level,
                report      = final_report,
                image_path  = alert_frame,
                location    = f"{location} / {time_of_day}",
                weather     = {
                    "wind_speed":  wind_speed,
                    "time_of_day": time_of_day,
                },
            )
        except Exception as tg_err:
            print(f"[Telegram] 전송 실패 (파이프라인은 계속): {tg_err}")

        return JSONResponse({
            "status":          "anomaly_detected",
            "location":        location,
            "time_of_day":     time_of_day,
            "anomaly_score":   round(anomaly_score, 2),
            "frames_analyzed": len(frame_paths),
            "weather": {
                "temperature": temperature,
                "wind_speed":  wind_speed,
                "humidity":    humidity,
                "rainfall":    rainfall,
            },
            "first_analysis": first_analysis,
            "first_level":    first_level,
            "similar_cases":  len(similar_cases),
            "final_report":   final_report,
            "final_level":    final_level,
            "level_changed":  first_level != final_level,
        })

    finally:
        # Clean up all temp files
        for fp in frame_paths:
            if os.path.exists(fp):
                os.unlink(fp)
        if os.path.exists(video_path):
            os.unlink(video_path)


# Keep the original image endpoint for backwards compatibility
@app.post("/analyze")
async def analyze(
    image:       UploadFile = File(...),
    location:    str        = Form(default="연평도"),
    time_of_day: str        = Form(default="주간"),
    temperature: float      = Form(default=20.0),
    wind_dir:    float      = Form(default=180.0),
    wind_speed:  float      = Form(default=5.0),
    rainfall:    float      = Form(default=0.0),
    humidity:    float      = Form(default=60.0),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        input_text = (
            f"해안 감시 카메라 이미지입니다. [{location} / {time_of_day}]\n"
            f"기상정보: 기온 {temperature}°C, 풍향 {wind_dir}°, "
            f"풍속 {wind_speed}m/s, 강수량 {rainfall}mm, 습도 {humidity}%\n\n"
            f"다음 형식으로 상황을 분석하고 대응 리포트를 작성하세요."
        )

        first_analysis = vlm_analyze([tmp_path], input_text)
        first_level    = extract_alert_level(first_analysis)

        object_type = None
        for obj in ["군함", "어선", "상선", "드론", "오물폭탄", "삐라", "항공기"]:
            if obj in first_analysis:
                object_type = obj
                break

        similar_cases = rag_search(first_analysis, object_type)
        case_texts    = "\n\n".join([
            f"[Case {i+1}] Alert:{doc.metadata['alert_level']} | "
            f"Object:{doc.metadata['object']} | "
            f"Time:{doc.metadata['time']}\n{doc.page_content[:200]}"
            for i, doc in enumerate(similar_cases)
        ])

        final_prompt = (
            f"해안 감시 카메라 이미지입니다. [{location} / {time_of_day}]\n"
            f"기상정보: 기온 {temperature}°C, 풍향 {wind_dir}°, "
            f"풍속 {wind_speed}m/s, 강수량 {rainfall}mm, 습도 {humidity}%\n\n"
            f"[1차 분석 결과]\n{first_analysis}\n\n"
            f"[유사 과거 사례]\n{case_texts}\n\n"
            f"1차 분석 결과와 유사 과거 사례를 참고하여 최종 대응 리포트를 작성하세요."
        )

        final_report  = vlm_analyze([tmp_path], final_prompt)
        final_level   = extract_alert_level(final_report)

        return JSONResponse({
            "status":         "success",
            "location":       location,
            "time_of_day":    time_of_day,
            "weather": {
                "temperature": temperature,
                "wind_speed":  wind_speed,
                "humidity":    humidity,
                "rainfall":    rainfall,
            },
            "first_analysis": first_analysis,
            "first_level":    first_level,
            "similar_cases":  len(similar_cases),
            "final_report":   final_report,
            "final_level":    final_level,
            "level_changed":  first_level != final_level,
        })

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
