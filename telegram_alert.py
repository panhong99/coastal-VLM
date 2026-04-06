# telegram_alert.py
import os
import requests
from datetime import datetime

TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_alert(alert_level, report, image_path, location, weather):
    """경보 레벨에 따라 텔레그램 전송 (GREEN은 전송 안 함)"""

    if alert_level not in ["YELLOW", "RED"]:
        return False

    emoji = {"YELLOW": "🟡", "RED": "🔴"}[alert_level]
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    message = f"""{emoji} [{alert_level} 경보] {location}
{'─' * 30}
📍 위치: {location}
🕐 시간: {now}
🌊 풍속: {weather.get('wind_speed', '-')}m/s
🌙 시간대: {weather.get('time_of_day', '-')}
{'─' * 30}
📋 상황 분석:
{report[:300]}
{'─' * 30}
⚠️ AI 자동 생성 보고서
최종 판단은 당직사령 확인 요망"""

    # 이미지 + 메시지 전송
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        res = requests.post(url, data={
            "chat_id": CHAT_ID,
            "caption": message,
        }, files={"photo": img})

    return res.ok