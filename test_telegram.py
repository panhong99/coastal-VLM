# test_telegram.py
# Run this first to verify Telegram bot is configured correctly,
# before testing the full pipeline.
#
# Usage:
#   export TELEGRAM_BOT_TOKEN="your_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"
#   python test_telegram.py

from telegram_alert import send_alert

send_alert(
    location    = "연평도 / 주간",
    object_type = "군함",
    report      = "test",
    frame_path  = "/home/hail/pan/VLM-project/dataset/open_data/data/Training/images/I1_images/I1_S0_C5_0008001.jpg"
)
