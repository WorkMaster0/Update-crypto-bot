import os
import logging
from flask import Flask, request
from telebot import TeleBot
from apscheduler.schedulers.background import BackgroundScheduler

# Власні модулі
from config import BOT_TOKEN, WEBHOOK_URL, ANALYSIS_SETTINGS, ALERT_SETTINGS
from utils.api_client import BinanceClient
from commands.basic_commands import setup_basic_commands
from commands.analysis_commands import setup_analysis_commands
from commands.alert_commands import setup_alert_commands

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація бота та Flask
bot = TeleBot(BOT_TOKEN)
app = Flask(__name__)
binance_client = BinanceClient()

# Налаштування команд
setup_basic_commands(bot, ANALYSIS_SETTINGS)
setup_analysis_commands(bot, ANALYSIS_SETTINGS)
setup_alert_commands(bot, ALERT_SETTINGS, binance_client)

# Webhook обробник
@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        update = request.get_json()
        bot.process_new_updates([update])
        return 'ok', 200

@app.route('/')
def index():
    return "Crypto Analysis Bot is running!"

# Запуск
if __name__ == "__main__":
    # Видаляємо старий вебхук
    bot.remove_webhook()
    
    # Встановлюємо новий вебхук
    if WEBHOOK_URL:
        bot.set_webhook(url=f"{WEBHOOK_URL}/webhook")
        logger.info(f"Webhook set to: {WEBHOOK_URL}/webhook")
    
    # Запускаємо Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)