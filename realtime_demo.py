# realtime_demo.py
import time
import requests
import gradio as gr
from telegram_alert import send_alert
import random
from pathlib import Path

API_URL = "http://localhost:8000/analyze"

def build_random_scenario(n=10):
    base = Path("/home/hail/pan/VLM-project/dataset")

    # 이미지 풀 수집
    real_images      = list((base / "open_data/data/Training/images/I1_images").glob("*.jpg"))
    synthetic_day    = list((base / "synthetic_data/new_dataset/open_data/data/Training/images/EO_SU_DT").glob("*.jpg"))
    synthetic_night  = list((base / "synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT").glob("*.jpg"))

    all_images = real_images + synthetic_day + synthetic_night

    # 기상 조건 랜덤 설정
    locations   = ["연평도", "백령도"]
    time_of_day = ["주간", "야간"]

    scenario = []
    for _ in range(n):
        img = random.choice(all_images)

        # 야간 이미지면 야간으로 설정
        is_night = "NT" in img.name or "야간" in img.name

        scenario.append({
            "image":       str(img),
            "location":    random.choice(locations),
            "time_of_day": "야간" if is_night else random.choice(time_of_day),
            "wind_speed":  round(random.uniform(0.5, 25.0), 1),
            "humidity":    round(random.uniform(40.0, 95.0), 1),
            "temperature": round(random.uniform(5.0, 35.0), 1),
            "rainfall":    round(random.uniform(0.0, 20.0), 1),
            "desc":        f"{'야간' if is_night else '주간'} 상황 분석 중"
        })

    return scenario

# 데모 시나리오
SCENARIO = [
    {
        "image":      "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0001001.jpg",
        "location":   "연평도",
        "time_of_day":"주간",
        "wind_speed":  2.0,
        "humidity":   60.0,
        "temperature":28.0,
        "rainfall":    0.0,
        "desc":       "정상 상황 - 주간 어선 항해 중"
    },
    {
        "image":      "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0120002.jpg",
        "location":   "연평도",
        "time_of_day":"야간",
        "wind_speed": 15.0,
        "humidity":   80.0,
        "temperature":15.0,
        "rainfall":    5.0,
        "desc":       "주의 상황 - 야간 군함 탐지"
    },
    {
        "image":      "/home/hail/pan/VLM-project/dataset/synthetic_data/new_dataset/open_data/data/Training/images/IR_SU_NT/IR_SU_NT_W1_H1_B1C1D1_0004.jpg",
        "location":   "백령도",
        "time_of_day":"야간",
        "wind_speed": 22.0,
        "humidity":   85.0,
        "temperature":10.0,
        "rainfall":   10.0,
        "desc":       "위협 상황 - 야간 강풍 군함"
    },
]

def run_demo():
    """데모 자동 실행 - Gradio generator"""

    RANDOM_SCENARIO = build_random_scenario(n=10)
    for i, sc in enumerate(RANDOM_SCENARIO):
        yield (
            sc["image"],
            f"""<div style="background:#1A3A4A;padding:15px;border-radius:8px;color:#00E5FF;">
                ⏳ {i+1}/{len(RANDOM_SCENARIO)} 분석 중... {sc['desc']}
            </div>""",
            "분석 중...",
            "",
        )

        # FastAPI 호출
        with open(sc["image"], "rb") as f:
            response = requests.post(API_URL, files={"image": f}, data={
                "location":    sc["location"],
                "time_of_day": sc["time_of_day"],
                "wind_speed":  sc["wind_speed"],
                "humidity":    sc["humidity"],
                "temperature": sc["temperature"],
                "rainfall":    sc["rainfall"],
            })

        result = response.json()
        level  = result["final_level"]

        # 텔레그램 전송
        telegram_sent = send_alert(
            alert_level=level,
            report=result["final_report"],
            image_path=sc["image"],
            location=sc["location"],
            weather=sc
        )

        # 경보 색상
        colors = {
            "GREEN":   "#00C853",
            "YELLOW":  "#FFD600",
            "RED":     "#D50000",
            "UNKNOWN": "#546E7A"   # ← 추가
        } 

        emoji = {
            "GREEN":   "🟢",
            "YELLOW":  "🟡",
            "RED":     "🔴",
            "UNKNOWN": "⚪"        # ← 추가
        }

        alert_html = f"""
        <div style="
            background:{colors[level]};
            color:white;
            padding:25px;
            border-radius:12px;
            text-align:center;
            font-size:2.2em;
            font-weight:bold;
            box-shadow:0 4px 15px rgba(0,0,0,0.3);
        ">
            {emoji[level]} {level} 경보
        </div>
        """

        telegram_status = ""
        if telegram_sent:
            telegram_status = "📲 텔레그램 전송 완료 ✅"
        else:
            telegram_status = "✅ 정상 상황 (전송 없음)"

        yield (
            sc["image"],
            alert_html,
            result["final_report"],
            telegram_status,
        )

        time.sleep(7)  # 7초 간격


# Gradio UI
with gr.Blocks(
    title="해안 경계 AI 실시간 모니터링",
    theme=gr.themes.Base(),
    css="""
    body { background-color: #0A1628; }
    .gradio-container { background-color: #0A1628; }
    """
) as demo:

    gr.HTML("""
    <div style="
        background:linear-gradient(135deg,#0A1628,#0D2137);
        color:white; padding:25px; border-radius:12px;
        text-align:center; margin-bottom:15px;
        border-left:5px solid #00E5FF;
    ">
        <h1 style="margin:0; font-size:1.8em;">
            🛡️ 해안 경계 AI 실시간 모니터링 시스템
        </h1>
        <p style="margin:8px 0 0 0; color:#B0BEC5;">
            Qwen2.5-VL 7B + RAG + 텔레그램 실시간 경보
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📹 실시간 CCTV 화면")
            cctv_display = gr.Image(
                label="현재 화면",
                height=320
            )
            telegram_status = gr.Textbox(
                label="텔레그램 전송 상태",
                interactive=False
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 AI 분석 결과")
            alert_display = gr.HTML(
                value="""<div style="
                    background:#1A3A4A; color:#B0BEC5;
                    padding:25px; border-radius:12px;
                    text-align:center; font-size:1.5em;
                ">⏳ 분석 대기 중</div>"""
            )
            report_output = gr.Markdown(
                value="데모를 시작하면 분석 결과가 여기에 표시됩니다."
            )

    start_btn = gr.Button(
        "▶  실시간 데모 시작",
        variant="primary",
        size="lg"
    )

    start_btn.click(
        fn=run_demo,
        outputs=[
            cctv_display,
            alert_display,
            report_output,
            telegram_status,
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)