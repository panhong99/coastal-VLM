# telegram_alert.py
import os
import re
import requests
from datetime import datetime

TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def _parse_field(report: str, field: str) -> str:
    """리포트 마크다운에서 **필드명:** 값을 추출."""
    match = re.search(rf"\*\*{field}:\*\*\s*(.+)", report)
    return match.group(1).strip() if match else "-"


def _clean_summary(summary: str) -> str:
    """날씨/계절/시간대 배경 묘사를 제거하고 탐지 내용만 반환."""
    # "낮 바다에", "야간 하늘에" 등 장소 표현 이후가 실제 탐지 내용
    match = re.search(r'(?:바다에|하늘에|해상에|공중에|수면에|해역에)\s*', summary)
    if match:
        return summary[match.end():].strip()
    return summary


def send_alert(alert_level, report, image_path, location, weather):
    """경보 레벨에 따라 텔레그램 전송 (GREEN은 전송 안 함)"""

    if alert_level not in ["YELLOW", "RED"]:
        return False

    if not TOKEN or not CHAT_ID:
        print("[Telegram] TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수가 설정되지 않음 — 전송 생략")
        return False

    emoji = {"YELLOW": "🟡", "RED": "🔴"}[alert_level]
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    object_type = _parse_field(report, "탐지 객체")
    summary     = _clean_summary(_parse_field(report, "상황 요약"))
    action      = _parse_field(report, "권고 조치")

    message = f"""{emoji} {alert_level} 경보 — {location}
🕐 {now}
{'─' * 28}
🎯 탐지 객체: {object_type}
⚠️ 경보 레벨: {alert_level}
{'─' * 28}
📋 {summary} {action}
{'─' * 28}
※ 최종 판단은 당직사령 확인 요망"""

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        res = requests.post(url, data={
            "chat_id": CHAT_ID,
            "caption": message,
        }, files={"photo": img})

    return res.ok