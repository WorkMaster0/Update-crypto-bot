# main.py
from flask import Flask, request
from app.bot import bot
import os

app = Flask(__name__)

# ---------- Настройки ----------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # повний URL: https://yourdomain.com/<token>

# ---------- Webhook Routes ----------
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    """Отримує всі оновлення від Telegram через webhook"""
    json_data = request.get_json(force=True)
    bot.process_new_updates([bot.types.Update.de_json(json_data)])
    return "OK", 200

@app.route("/")
def index():
    return "Bot is running!", 200

# ---------- Установка Webhook ----------
def set_webhook():
    if not WEBHOOK_URL:
        raise RuntimeError("WEBHOOK_URL is not set")
    success = bot.set_webhook(url=WEBHOOK_URL)
    if success:
        print(f"Webhook встановлено: {WEBHOOK_URL}")
    else:
        print("Помилка встановлення webhook")

# ---------- Запуск Flask ----------
if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))