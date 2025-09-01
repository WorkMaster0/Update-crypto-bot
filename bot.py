import os
import requests
import logging
import threading
import time
from datetime import datetime
from flask import Flask, request
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler

# Налаштування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("BOT_TOKEN не знайдено в змінних оточення")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# Глобальні змінні для налаштувань
USER_SETTINGS = {
    'min_volume': 5000000,
    'top_symbols': 30,
    'window_size': 20,
    'sensitivity': 0.005,
    'pump_threshold': 15,  # % для виявлення пампу
    'dump_threshold': -15  # % для виявлення дампу
}

# Словник для зберігання чатів, які підписані на сповіщення
ALERT_SUBSCRIPTIONS = {}

# Допоміжні функції
def get_klines(symbol, interval="1h", limit=200):
    """Отримання історичних даних з Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        data = requests.get(url, params=params, timeout=10).json()
        
        if not data:
            return None
            
        df = {
            'o': [float(c[1]) for c in data],
            'h': [float(c[2]) for c in data],
            'l': [float(c[3]) for c in data],
            'c': [float(c[4]) for c in data],
            'v': [float(c[5]) for c in data],
            't': [c[0] for c in data]
        }
        return df
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
        return None

def find_support_resistance(prices, window=20, delta=0.005):
    """Знаходження рівнів підтримки та опору"""
    n = len(prices)
    rolling_high = [0] * n
    rolling_low = [0] * n
    
    # Заповнюємо rolling_high та rolling_low
    for i in range(window, n):
        rolling_high[i] = max(prices[i-window:i])
        rolling_low[i] = min(prices[i-window:i])
    
    levels = []
    for i in range(window, n):
        if prices[i] >= rolling_high[i] * (1 - delta):
            levels.append(rolling_high[i])
        elif prices[i] <= rolling_low[i] * (1 + delta):
            levels.append(rolling_low[i])
    
    return sorted(set(levels))

def calculate_rsi(prices, period=14):
    """Розрахунок RSI без numpy"""
    if len(prices) < period + 1:
        return 50  # Недостатньо даних
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_volume_spike(volumes, lookback=20):
    """Перевірка сплеску обсягів"""
    if len(volumes) < lookback:
        return False
    recent_volume = volumes[-1]
    avg_volume = sum(volumes[-lookback:]) / lookback
    return recent_volume > 1.5 * avg_volume

def calculate_technical_indicators(closes, volumes):
    """Розрахунок технічних індикаторів"""
    rsi = calculate_rsi(closes)
    vol_spike = calculate_volume_spike(volumes)
    return rsi, vol_spike

def detect_pump_dump(closes, volumes, pump_threshold=15, dump_threshold=-15):
    """Виявлення пампу або дампу"""
    if len(closes) < 24:
        return None, 0
    
    # Зміна ціни за останні 24 години
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    
    # Перевірка обсягу
    vol_spike = calculate_volume_spike(volumes)
    
    # Визначення типу події
    event_type = None
    if price_change_24h > pump_threshold and vol_spike:
        event_type = "PUMP"
    elif price_change_24h < dump_threshold and vol_spike:
        event_type = "DUMP"
    
    return event_type, price_change_24h

def analyze_market():
    """Аналіз ринку для пошуку пампів/дампів"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:USER_SETTINGS['top_symbols']]]

        alerts = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]

                # Перевірка на памп/дамп
                event_type, price_change = detect_pump_dump(
                    closes, volumes, 
                    USER_SETTINGS['pump_threshold'], 
                    USER_SETTINGS['dump_threshold']
                )

                if event_type:
                    alert_text = (
                        f"🔴 {event_type} DETECTED!\n"
                        f"Токен: {symbol}\n"
                        f"Зміна ціни: {price_change:+.1f}%\n"
                        f"Рекомендація: {'Шорт' if event_type == 'PUMP' else 'Лонг'}"
                    )
                    alerts.append(alert_text)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        return alerts

    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return []

def send_alerts_to_subscribers():
    """Надсилання сповіщень всім підписникам"""
    if not ALERT_SUBSCRIPTIONS:
        return
    
    alerts = analyze_market()
    if not alerts:
        return
    
    alert_text = "\n\n".join(alerts[:5])  # Обмежуємо кількість сповіщень
    
    for chat_id in ALERT_SUBSCRIPTIONS.keys():
        try:
            bot.send_message(chat_id, f"🚨 АВТОМАТИЧНЕ СПОВІЩЕННЯ:\n\n{alert_text}")
        except Exception as e:
            logger.error(f"Error sending alert to {chat_id}: {e}")

# Планувальник для автоматичних перевірок
scheduler = BackgroundScheduler()
scheduler.add_job(send_alerts_to_subscribers, 'interval', minutes=30)
scheduler.start()

# Flask маршрути для вебхуків
@app.route('/')
def index():
    return "Crypto Bot is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        update = telebot.types.Update.de_json(request.stream.read().decode('utf-8'))
        bot.process_new_updates([update])
        return 'ok', 200

# Команди бота
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Привітальне повідомлення"""
    help_text = """
🤖 Smart Crypto Bot - Аналіз пампів та дампів

Доступні команди:
/smart_auto - Автоматичний пошук сигналів
/settings - Налаштування параметрів
/check_token <token> - Перевірити конкретний токен
/stats - Статистика ринку
/alerts_on - Увімкнути сповіщення
/alerts_off - Вимкнути сповіщення
"""
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['alerts_on'])
def enable_alerts(message):
    """Увімкнути сповіщення"""
    ALERT_SUBSCRIPTIONS[message.chat.id] = True
    bot.reply_to(message, "🔔 Сповіщення увімкнено! Ви отримуватимете автоматичні сповіщення про памп/дамп.")

@bot.message_handler(commands=['alerts_off'])
def disable_alerts(message):
    """Вимкнути сповіщення"""
    if message.chat.id in ALERT_SUBSCRIPTIONS:
        del ALERT_SUBSCRIPTIONS[message.chat.id]
    bot.reply_to(message, "🔕 Сповіщення вимкнено.")

@bot.message_handler(commands=['settings'])
def show_settings(message):
    """Налаштування параметрів"""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    keyboard.add(
        KeyboardButton("Мін. обсяг 📊"),
        KeyboardButton("Кількість монет 🔢"),
        KeyboardButton("Чутливість ⚖️"),
        KeyboardButton("PUMP % 📈"),
        KeyboardButton("DUMP % 📉"),
        KeyboardButton("Головне меню 🏠")
    )
    
    settings_text = f"""
Поточні налаштування:

Мінімальний обсяг: {USER_SETTINGS['min_volume']:,.0f} USDT
Кількість монет для аналізу: {USER_SETTINGS['top_symbols']}
Чутливість: {USER_SETTINGS['sensitivity'] * 100}%
PUMP поріг: {USER_SETTINGS['pump_threshold']}%
DUMP поріг: {USER_SETTINGS['dump_threshold']}%
"""
    bot.send_message(message.chat.id, settings_text, reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text == "Мін. обсяг 📊")
def set_min_volume(message):
    msg = bot.send_message(message.chat.id, "Введіть мінімальний обсяг торгів (USDT):")
    bot.register_next_step_handler(msg, process_min_volume)

def process_min_volume(message):
    try:
        volume = float(message.text.replace(',', '').replace(' ', ''))
        USER_SETTINGS['min_volume'] = volume
        bot.send_message(message.chat.id, f"Мінімальний обсяг встановлено: {volume:,.0f} USDT")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "Кількість монет 🔢")
def set_top_symbols(message):
    msg = bot.send_message(message.chat.id, "Введіть кількість монет для аналізу:")
    bot.register_next_step_handler(msg, process_top_symbols)

def process_top_symbols(message):
    try:
        count = int(message.text)
        USER_SETTINGS['top_symbols'] = count
        bot.send_message(message.chat.id, f"Кількість монет для аналізу встановлено: {count}")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть ціле число.")

@bot.message_handler(func=lambda message: message.text == "Чутливість ⚖️")
def set_sensitivity(message):
    msg = bot.send_message(message.chat.id, "Введіть чутливість (0.1-5.0%):")
    bot.register_next_step_handler(msg, process_sensitivity)

def process_sensitivity(message):
    try:
        sensitivity = float(message.text)
        if 0.1 <= sensitivity <= 5.0:
            USER_SETTINGS['sensitivity'] = sensitivity / 100
            bot.send_message(message.chat.id, f"Чутливість встановлено: {sensitivity}%")
        else:
            bot.send_message(message.chat.id, "❌ Значення повинно бути між 0.1 та 5.0")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "PUMP % 📈")
def set_pump_threshold(message):
    msg = bot.send_message(message.chat.id, "Введіть поріг для виявлення PUMP (%):")
    bot.register_next_step_handler(msg, process_pump_threshold)

def process_pump_threshold(message):
    try:
        threshold = float(message.text)
        USER_SETTINGS['pump_threshold'] = threshold
        bot.send_message(message.chat.id, f"PUMP поріг встановлено: {threshold}%")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "DUMP % 📉")
def set_dump_threshold(message):
    msg = bot.send_message(message.chat.id, "Введіть поріг для виявлення DUMP (%):")
    bot.register_next_step_handler(msg, process_dump_threshold)

def process_dump_threshold(message):
    try:
        threshold = float(message.text)
        USER_SETTINGS['dump_threshold'] = threshold
        bot.send_message(message.chat.id, f"DUMP поріг встановлено: {threshold}%")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "Головне меню 🏠")
def main_menu(message):
    send_welcome(message)

@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    """Основна функція пошуку сигналів"""
    try:
        msg = bot.send_message(message.chat.id, "🔍 Аналізую ринок...")
        
        alerts = analyze_market()
        if alerts:
            alert_text = "\n\n".join(alerts[:10])
            bot.edit_message_text(f"<b>🚨 Знайдено сигнали:</b>\n\n{alert_text}", 
                                 message.chat.id, msg.message_id, parse_mode="HTML")
        else:
            bot.edit_message_text("ℹ️ Жодних сигналів не знайдено.", 
                                 message.chat.id, msg.message_id)

    except Exception as e:
        logger.error(f"Error in smart_auto: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['check_token'])
def check_token_handler(message):
    """Перевірка конкретного токена"""
    try:
        symbol = message.text.split()[1].upper() + "USDT"
        df = get_klines(symbol, interval="1h", limit=200)
        
        if not df:
            bot.send_message(message.chat.id, "❌ Токен не знайдено або помилка даних")
            return
            
        closes = [float(c) for c in df["c"]]
        volumes = [float(v) for v in df["v"]]
        last_price = closes[-1]
        
        # Аналіз
        rsi, vol_spike = calculate_technical_indicators(closes, volumes)
        sr_levels = find_support_resistance(closes)
        event_type, price_change = detect_pump_dump(closes, volumes)
        
        analysis_text = f"""
<b>{symbol} Analysis</b>

Поточна ціна: ${last_price:.4f}
RSI: {rsi:.1f} {'(перекупленість)' if rsi > 70 else '(перепроданість)' if rsi < 30 else ''}
Обсяг: {'підвищений' if vol_spike else 'нормальний'}
Подія: {event_type if event_type else 'немає'} ({price_change:+.1f}%)

<b>Key Levels:</b>
"""
        for level in sr_levels[-5:]:  # Останні 5 рівнів
            distance_pct = (last_price - level) / level * 100
            analysis_text += f"{level:.4f} ({distance_pct:+.1f}%)\n"

        # Додаємо рекомендацію
        if event_type == "PUMP":
            analysis_text += "\n🔴 Рекомендація: Шорт (можливий корекція після пампу)"
        elif event_type == "DUMP":
            analysis_text += "\n🟢 Рекомендація: Лонг (можливий відскок після дампу)"

        bot.send_message(message.chat.id, analysis_text, parse_mode="HTML")
        
    except IndexError:
        bot.send_message(message.chat.id, "ℹ️ Використання: /check_token BTC")
    except Exception as e:
        logger.error(f"Error in check_token: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['stats'])
def market_stats(message):
    """Статистика ринку"""
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()
        
        # Фільтруємо USDT пари з високим обсягом
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 1000000]
        
        # Топ гейнери/лосери
        gainers = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']), reverse=True)[:5]
        losers = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']))[:5]
        
        stats_text = "<b>📈 Market Statistics</b>\n\n"
        stats_text += "<b>Top Gainers:</b>\n"
        for i, coin in enumerate(gainers, 1):
            stats_text += f"{i}. {coin['symbol']} +{float(coin['priceChangePercent']):.1f}%\n"
        
        stats_text += "\n<b>Top Losers:</b>\n"
        for i, coin in enumerate(losers, 1):
            stats_text += f"{i}. {coin['symbol']} {float(coin['priceChangePercent']):.1f}%\n"
            
        bot.send_message(message.chat.id, stats_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in stats: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

if __name__ == "__main__":
    # Отримуємо порт з оточення (потрібно для Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Видаляємо вебхук якщо він був встановлений раніше
    bot.remove_webhook()
    
    # Встановлюємо вебхук
    webhook_url = os.environ.get('WEBHOOK_URL', '')
    if webhook_url:
        bot.set_webhook(url=f"{webhook_url}/webhook")
        logger.info(f"Webhook set to: {webhook_url}/webhook")
        app.run(host='0.0.0.0', port=port)
    else:
        # Якщо WEBHOOK_URL не встановлено, використовуємо polling
        logger.info("Starting bot with polling...")
        bot.polling(none_stop=True)