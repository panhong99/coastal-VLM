# gradio_ui.py
import gradio as gr
import requests
from datetime import datetime

API_URL = "http://localhost:8000/analyze_video"

# Pipeline step labels shown in the UI
PIPELINE_STEPS = [
    "① CCTV Monitoring",
    "② Anomaly Detection",
    "③ VLM First Analysis",
    "④ RAG Case Search",
    "⑤ Report Generation",
]


def build_step_html(active_step: int, anomaly_detected: bool = True) -> str:
    """
    Render the 5-step pipeline progress bar as HTML.
    active_step: 0-indexed step that is currently completed (0 = just started).
    """
    colors = {
        "done":    "#2ecc71",
        "active":  "#e67e22",
        "pending": "#555555",
        "skip":    "#888888",
    }

    boxes = []
    for i, label in enumerate(PIPELINE_STEPS):
        if i < active_step:
            color = colors["done"]
            icon  = "✔"
        elif i == active_step:
            color = colors["active"]
            icon  = "▶"
        else:
            color = colors["pending"]
            icon  = "○"

        # If no anomaly was detected, grey out steps 3-5
        if not anomaly_detected and i >= 2:
            color = colors["skip"]
            icon  = "–"

        boxes.append(f"""
        <div style="
            background:{color}; color:white;
            padding:12px 8px; border-radius:8px;
            text-align:center; font-size:0.82em; font-weight:bold;
            min-width:110px; flex:1;
        ">
            <div style="font-size:1.1em">{icon}</div>
            {label}
        </div>
        """)

    arrow = '<div style="color:#aaa; font-size:1.4em; align-self:center;">→</div>'
    inner = arrow.join(boxes)

    return f"""
    <div style="
        display:flex; gap:6px; align-items:stretch;
        background:#1a1a2e; padding:14px; border-radius:10px;
    ">
        {inner}
    </div>
    """


def analyze_video(
    video,
    location,
    time_of_day,
    temperature,
    wind_dir,
    wind_speed,
    rainfall,
    humidity,
):
    if video is None:
        return (
            build_step_html(0),
            "Please upload a video file.",
            "",
            "",
        )

    try:
        # ── Step 1 → 2: send video to API ────────────────────────────────────
        with open(video, "rb") as f:
            files = {"video": f}
            data  = {
                "location":    location,
                "time_of_day": time_of_day,
                "temperature": temperature,
                "wind_dir":    wind_dir,
                "wind_speed":  wind_speed,
                "rainfall":    rainfall,
                "humidity":    humidity,
            }
            response = requests.post(API_URL, files=files, data=data, timeout=300)
            result   = response.json()

        # ── No motion detected (pixel diff below threshold) ──────────────────
        if result["status"] == "no_anomaly":
            step_html = build_step_html(1, anomaly_detected=False)
            banner = """
            <div style="
                background:#1e7e34; color:white;
                padding:20px; border-radius:10px;
                text-align:center; font-size:1.5em; font-weight:bold;
            ">
                ✅ No Motion Detected — Continuing Surveillance
            </div>
            """
            info = (
                f"**Anomaly score:** {result['anomaly_score']}  \n"
                f"**Frames analyzed:** {result['frames_analyzed']}  \n"
                f"**Threshold:** 15.0"
            )
            return step_html, banner, info, ""

        # ── Motion detected but no threat (VLM returned GREEN) ───────────────
        if result["status"] == "no_threat":
            # Pipeline ran through step 3 (VLM), stopped before RAG/report
            step_html = build_step_html(2, anomaly_detected=True)
            banner = f"""
            <div style="
                background:#1e7e34; color:white;
                padding:20px; border-radius:10px;
                text-align:center; font-size:1.5em; font-weight:bold;
            ">
                🟢 Object Detected — No Threat (GREEN) — Continuing Surveillance
            </div>
            """
            info = (
                f"**VLM result:** {result['first_level']}  \n"
                f"**Detected object:** {result.get('object_type') or 'unknown'}  \n"
                f"**Anomaly score:** {result['anomaly_score']}  \n\n"
                f"{result['first_analysis']}"
            )
            return step_html, banner, info, ""

        # ── Anomaly detected: full pipeline completed ─────────────────────────
        if result["status"] != "anomaly_detected":
            return build_step_html(0), "Analysis failed.", "", ""

        level       = result["final_level"]
        level_emoji = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(level, "⚪")
        level_color = {
            "RED":    "#c0392b",
            "YELLOW": "#d68910",
            "GREEN":  "#1e8449",
        }.get(level, "#888888")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_md = f"""
# {level_emoji} Coastal Surveillance Situation Report

---

### Report Info
| Field | Value |
|---|---|
| Reported at | {now} |
| Location | {location} |
| Condition | {time_of_day} |
| Alert level | **{level_emoji} {level}** |
| Frames analyzed | {result['frames_analyzed']} |
| Anomaly score | {result['anomaly_score']} |
| RAG cases referenced | {result['similar_cases']} |

---

### Weather
| Item | Value |
|---|---|
| Temperature | {temperature}°C |
| Wind direction | {wind_dir}° |
| Wind speed | {wind_speed} m/s |
| Rainfall | {rainfall} mm |
| Humidity | {humidity}% |

---

### AI Analysis Result

{result['final_report']}

---

### Analysis Process
- **VLM first alert level:** {result['first_level']}
- **RAG cases referenced:** {result['similar_cases']}
- **Final alert level:** {level}
{"- ⚠️ **RAG correction applied:** " + result['first_level'] + " → " + level if result['level_changed'] else "- ✅ **Consistent with first analysis**"}

---
*This report was automatically generated by the AI system.*
*Final judgment requires confirmation by the duty officer.*
"""

        alert_html = f"""
        <div style="
            background:{level_color}; color:white;
            padding:20px; border-radius:10px;
            text-align:center; font-size:2em; font-weight:bold;
        ">
            {level_emoji} Alert Level: {level}
        </div>
        """

        first_md = f"### VLM First Analysis\n{result['first_analysis']}"

        return build_step_html(4), alert_html, report_md, first_md

    except Exception as e:
        return build_step_html(0), f"Error: {str(e)}", "", ""


# ── Gradio UI layout ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="Coastal Surveillance AI",
    theme=gr.themes.Base(),
    css="""
    .container { max-width: 1300px; margin: auto; }
    .header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white; padding: 20px; border-radius: 10px;
        margin-bottom: 16px; text-align: center;
    }
    """
) as demo:

    # Header
    gr.HTML("""
    <div class="header">
        <h1>🛡️ Coastal Surveillance AI System</h1>
        <p>Real-time Event-Driven Pipeline: Video → Anomaly Detection → VLM → RAG → Report</p>
        <p style="font-size:0.8em; opacity:0.7;">
            Qwen2.5-VL 7B LoRA | ChromaDB RAG | FastAPI
        </p>
    </div>
    """)

    # Pipeline step progress bar (updates dynamically)
    pipeline_display = gr.HTML(value=build_step_html(0))

    with gr.Row():
        # ── Input panel ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📹 Input")

            video_input = gr.Video(
                label="Coastal CCTV Video",
                height=280,
            )

            with gr.Group():
                gr.Markdown("#### Location Info")
                location = gr.Dropdown(
                    choices=["연평도", "백령도", "기타"],
                    value="연평도",
                    label="Location"
                )
                time_of_day = gr.Radio(
                    choices=["주간", "야간"],
                    value="주간",
                    label="Time of day"
                )

            with gr.Group():
                gr.Markdown("#### Weather Info")
                with gr.Row():
                    temperature = gr.Number(value=20.0, label="Temp (°C)")
                    wind_dir    = gr.Number(value=180.0, label="Wind dir (°)")
                with gr.Row():
                    wind_speed = gr.Number(value=5.0,  label="Wind speed (m/s)")
                    rainfall   = gr.Number(value=0.0,  label="Rainfall (mm)")
                humidity = gr.Slider(
                    minimum=0, maximum=100, value=60, label="Humidity (%)"
                )

            analyze_btn = gr.Button("🔍 Start Analysis", variant="primary", size="lg")

        # ── Output panel ──────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Result")

            # Alert level (large display)
            alert_display = gr.HTML(value="""
            <div style="
                background:#555; color:white;
                padding:20px; border-radius:10px;
                text-align:center; font-size:2em;
            ">⏳ Waiting for analysis</div>
            """)

            with gr.Tabs():
                with gr.Tab("📋 Situation Report"):
                    report_output = gr.Markdown(
                        value="Upload a video and start analysis."
                    )
                with gr.Tab("🔍 VLM First Analysis"):
                    first_output = gr.Markdown()

    # Demo video examples
    gr.Markdown("### 📁 Demo Videos")
    gr.Examples(
        examples=[
            ["/home/hail/pan/VLM-project/demo_videos/green_01_어선_주간_기상양호.mp4",
             "연평도", "주간", 24.0, 180.0, 3.0, 0.0, 60],
            ["/home/hail/pan/VLM-project/demo_videos/green_02_어선_주간_기상양호.mp4",
             "연평도", "주간", 22.0, 200.0, 4.0, 0.0, 55],
            ["/home/hail/pan/VLM-project/demo_videos/green_03_상선_주간_기상양호.mp4",
             "백령도", "주간", 20.0, 160.0, 5.0, 0.0, 65],
            ["/home/hail/pan/VLM-project/demo_videos/yellow_01_군함_주간.mp4",
             "연평도", "주간", 18.0, 270.0, 8.0, 0.0, 70],
            ["/home/hail/pan/VLM-project/demo_videos/yellow_02_군함_주간_바람강함.mp4",
             "백령도", "주간", 16.0, 290.0, 14.0, 2.0, 75],
            ["/home/hail/pan/VLM-project/demo_videos/yellow_03_군함_야간.mp4",
             "연평도", "야간", 14.0, 260.0, 10.0, 1.0, 80],
            ["/home/hail/pan/VLM-project/demo_videos/red_01_드론_주간_합성.mp4",
             "연평도", "주간", 28.0, 180.0, 2.0, 0.0, 60],
            ["/home/hail/pan/VLM-project/demo_videos/red_02_군함_야간_풍랑.mp4",
             "백령도", "야간", 12.0, 310.0, 18.0, 8.0, 85],
            ["/home/hail/pan/VLM-project/demo_videos/red_03_드론_주간_합성2.mp4",
             "연평도", "주간", 26.0, 190.0, 3.0, 0.0, 62],
        ],
        inputs=[
            video_input, location, time_of_day,
            temperature, wind_dir, wind_speed, rainfall, humidity
        ]
    )

    # Wire up the button
    analyze_btn.click(
        fn=analyze_video,
        inputs=[
            video_input, location, time_of_day,
            temperature, wind_dir, wind_speed, rainfall, humidity
        ],
        outputs=[pipeline_display, alert_display, report_output, first_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
