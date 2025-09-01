# main.py
from flask import Flask, request
from telebot import TeleBot, types
from app.config import TELEGRAM_BOT_TOKEN
from app.bot import bot
from app.handlers import ai_alert_handler, ai_notify_handler
from app.charts import plot_candles
import io

app = Flask(__name__)

# ---------- TELEGRAM WEBHOOK ----------
@app.route(f"/webhook/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    json_data = request.get_json(force=True)
    bot.process_new_updates([types.Update.de_json(json_data)])
    return "OK", 200

# ---------- TEST GRAPH ENDPOINT ----------
@app.route("/chart/<symbol>")
def get_chart(symbol):
    buf = plot_candles(symbol.upper(), interval="1h", limit=200)
    return app.response_class(buf.getvalue(), mimetype='image/png')

# ---------- POLLING MODE (LOKALЬНЕ ТЕСТУВАННЯ) ----------
if __name__ == "__main__":
    # Якщо локально, зручно тестувати через polling
    print("Запуск бота в режимі polling...")
    bot.infinity_polling()