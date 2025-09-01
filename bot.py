from arbitrage_analyzer import arbitrage_analyzer
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
    'pump_threshold': 15,
    'dump_threshold': -15,
    'volume_spike_multiplier': 2.0,
    'rsi_overbought': 70,
    'rsi_oversold': 30
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
    """Розрахунок RSI"""
    if len(prices) < period + 1:
        return 50
    
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
    return recent_volume > USER_SETTINGS['volume_spike_multiplier'] * avg_volume

def calculate_technical_indicators(closes, volumes):
    """Розрахунок технічних індикаторів"""
    rsi = calculate_rsi(closes)
    vol_spike = calculate_volume_spike(volumes)
    return rsi, vol_spike

def detect_pump_dump(closes, volumes, pump_threshold=15, dump_threshold=-15):
    """Виявлення пампу або дампу"""
    if len(closes) < 24:
        return None, 0
    
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    vol_spike = calculate_volume_spike(volumes)
    
    event_type = None
    if price_change_24h > pump_threshold and vol_spike:
        event_type = "PUMP"
    elif price_change_24h < dump_threshold and vol_spike:
        event_type = "DUMP"
    
    return event_type, price_change_24h

def detect_pump_activity(symbol, closes, volumes, settings):
    """Розширений детектор памп-активності"""
    if len(closes) < 24:
        return None, 0, {}
    
    # Основні метрики
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    price_change_1h = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
    
    # Аналіз обсягів
    volume_metrics = analyze_volume(volumes, settings)
    
    # Додаткові показники
    volatility = calculate_volatility(closes[-24:])
    green_candles = count_green_candles(closes[-24:])
    
    # Визначення пампу
    is_pump = (
        price_change_24h > settings['pump_threshold'] and
        volume_metrics['volume_spike'] and
        price_change_1h > 5 and  # Різкий зліт за останню годину
        green_candles > 15  # Більшість свічок зростаючі
    )
    
    if not is_pump:
        return None, price_change_24h, volume_metrics
    
    # Рівень ризику (1-10)
    risk_level = calculate_pump_risk(closes, volumes, price_change_24h)
    
    pump_data = {
        'risk_level': risk_level,
        '1h_change': price_change_1h,
        'volatility': volatility,
        'green_candles': green_candles,
        'volume_metrics': volume_metrics
    }
    
    return "PUMP", price_change_24h, pump_data

def analyze_volume(volumes, settings):
    """Детальний аналіз обсягів торгів"""
    if len(volumes) < 24:
        return {'volume_spike': False, 'avg_volume': 0}
    
    current_volume = volumes[-1]
    avg_volume_24h = sum(volumes[-24:]) / 24
    avg_volume_7d = sum(volumes[-168:]) / 168 if len(volumes) >= 168 else avg_volume_24h
    
    volume_spike = current_volume > avg_volume_24h * settings['volume_spike_multiplier']
    volume_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 0
    
    return {
        'volume_spike': volume_spike,
        'avg_volume_24h': avg_volume_24h,
        'avg_volume_7d': avg_volume_7d,
        'volume_ratio': volume_ratio,
        'current_volume': current_volume
    }

def calculate_volatility(prices):
    """Розрахунок волатильності"""
    if len(prices) < 2:
        return 0
    
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    return sum(abs(r) for r in returns) / len(returns) * 100

def count_green_candles(prices):
    """Підрахунок зростаючих свічок"""
    if len(prices) < 2:
        return 0
    
    green_count = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            green_count += 1
    
    return green_count

def calculate_pump_risk(closes, volumes, price_change):
    """Розрахунок рівня ризику пампу"""
    risk = 5  # Базовий рівень
    
    # Корекція на основі величини зростання
    if price_change > 50:
        risk += 3
    elif price_change > 30:
        risk += 2
    elif price_change > 15:
        risk += 1
    
    # Корекція на основі обсягів
    if len(volumes) > 0:
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if sum(volumes[-10:]) > 0 else 1
        if volume_ratio > 5:
            risk += 2
        elif volume_ratio > 3:
            risk += 1
    
    # Обмеження від 1 до 10
    return max(1, min(10, risk))

def detect_volume_anomaly(symbol, volumes, settings):
    """Виявлення аномальних обсягів торгів"""
    if len(volumes) < 24:
        return False, {}
    
    current_volume = volumes[-1]
    avg_volume_24h = sum(volumes[-24:]) / 24
    
    # Перевірка на аномальний обсяг
    is_anomaly = current_volume > avg_volume_24h * settings['volume_spike_multiplier'] * 1.5
    
    if not is_anomaly:
        return False, {}
    
    anomaly_data = {
        'current_volume': current_volume,
        'avg_volume_24h': avg_volume_24h,
        'volume_ratio': current_volume / avg_volume_24h,
        'anomaly_type': 'VOLUME_SPIKE'
    }
    
    return True, anomaly_data

def send_alerts_to_subscribers():
    """Надсилання сповіщень всім підписникам"""
    if not ALERT_SUBSCRIPTIONS:
        return
    
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x.get("priceChangePercent", 0))),
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
                event_type, price_change = detect_pump_dump(closes, volumes)
                
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

        if alerts:
            alert_text = "\n\n".join(alerts[:3])  # Обмежуємо кількість сповіщень
            
            for chat_id in ALERT_SUBSCRIPTIONS.keys():
                try:
                    bot.send_message(chat_id, f"🚨 АВТОМАТИЧНЕ СПОВІЩЕННЯ:\n\n{alert_text}")
                except Exception as e:
                    logger.error(f"Error sending alert to {chat_id}: {e}")
                    
    except Exception as e:
        logger.error(f"Error in alert system: {e}")

# Планувальник для автоматичних перевірок
scheduler = BackgroundScheduler()
scheduler.add_job(send_alerts_to_subscribers, 'interval', minutes=30)
scheduler.start()

# Flask маршрути
@app.route('/')
def index():
    return "Crypto Bot is running!"

# Команди бота
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Привітальне повідомлення"""
    help_text = """
🤖 Smart Crypto Bot - Аналіз пампів та дампів

Доступні команди:
/smart_auto - Автоматичний пошук сигналів
/pump_scan - Сканування на памп активність
/volume_anomaly - Пошук аномальних обсягів
/advanced_analysis <token> - Розширений аналіз токена
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

@bot.message_handler(commands=['pump_scan'])
def pump_scan_handler(message):
    """Сканування на памп активність"""
    try:
        msg = bot.send_message(message.chat.id, "🔍 Сканую на памп активність...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and 
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x.get("priceChangePercent", 0))),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:USER_SETTINGS['top_symbols']]]
        pump_signals = []
        
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue
                
                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]
                
                # Детектуємо памп активність
                pump_type, price_change, pump_data = detect_pump_activity(
                    symbol, closes, volumes, USER_SETTINGS
                )
                
                if pump_type == "PUMP":
                    risk_level = pump_data.get('risk_level', 5)
                    risk_emoji = "🔴" if risk_level > 7 else "🟡" if risk_level > 5 else "🟢"
                    
                    signal_text = (
                        f"{risk_emoji} <b>{symbol}</b>\n"
                        f"📈 Зміна ціни: {price_change:+.1f}%\n"
                        f"⚠️ Рівень ризику: {risk_level}/10\n"
                        f"📊 Волатильність: {pump_data.get('volatility', 0):.1f}%\n"
                        f"🟢 Зелені свічки: {pump_data.get('green_candles', 0)}/24\n"
                        f"💹 Співвідношення обсягу: {pump_data.get('volume_metrics', {}).get('volume_ratio', 0):.1f}x\n"
                    )
                    
                    if risk_level > 7:
                        signal_text += "🔻 Високий ризик корекції!\n"
                    
                    pump_signals.append(signal_text)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not pump_signals:
            bot.edit_message_text("ℹ️ Пампи не знайдено.", message.chat.id, msg.message_id)
        else:
            text = "<b>🚨 Результати сканування пампа:</b>\n\n" + "\n".join(pump_signals[:5])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in pump_scan: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['volume_anomaly'])
def volume_anomaly_handler(message):
    """Пошук аномальних обсягів торгів"""
    try:
        msg = bot.send_message(message.chat.id, "🔍 Шукаю аномальні обсяги...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and 
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume'] / 10
        ]

        symbols = sorted(
            symbols,
            key=lambda x: float(x.get("quoteVolume", 0)),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:50]]
        anomalies = []
        
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=100)
                if not df or len(df.get("v", [])) < 24:
                    continue
                
                volumes = [float(v) for v in df["v"]]
                
                # Шукаємо аномалії обсягу
                is_anomaly, anomaly_data = detect_volume_anomaly(symbol, volumes, USER_SETTINGS)
                
                if is_anomaly:
                    anomaly_text = (
                        f"📊 <b>{symbol}</b>\n"
                        f"💥 Поточний обсяг: {anomaly_data.get('current_volume', 0):.0f}\n"
                        f"📈 Середній обсяг: {anomaly_data.get('avg_volume_24h', 0):.0f}\n"
                        f"🚀 Співвідношення: {anomaly_data.get('volume_ratio', 0):.1f}x\n"
                    )
                    anomalies.append(anomaly_text)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not anomalies:
            bot.edit_message_text("ℹ️ Аномалій обсягу не знайдено.", message.chat.id, msg.message_id)
        else:
            text = "<b>📈 Аномальні обсяги торгів:</b>\n\n" + "\n".join(anomalies[:8])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in volume_anomaly: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['advanced_analysis'])
def advanced_analysis_handler(message):
    """Розширений аналіз обраного токена"""
    try:
        # Перевіряємо, чи вказано токен
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /advanced_analysis BTC")
            return
            
        symbol = parts[1].upper() + "USDT"
        msg = bot.send_message(message.chat.id, f"🔍 Аналізую {symbol}...")
        
        # Отримуємо дані
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df.get("c", [])) < 50:
            bot.edit_message_text("❌ Не вдалося отримати дані для цього токена", message.chat.id, msg.message_id)
            return
        
        closes = [float(c) for c in df["c"]]
        volumes = [float(v) for v in df["v"]]
        last_price = closes[-1]
        
        # Виконуємо різні види аналізу
        pump_type, price_change, pump_data = detect_pump_activity(symbol, closes, volumes, USER_SETTINGS)
        is_volume_anomaly, volume_data = detect_volume_anomaly(symbol, volumes, USER_SETTINGS)
        volume_metrics = analyze_volume(volumes, USER_SETTINGS)
        
        # Формуємо звіт
        report_text = f"<b>📊 Розширений аналіз {symbol}</b>\n\n"
        report_text += f"💰 Поточна ціна: ${last_price:.4f}\n"
        report_text += f"📈 Зміна за 24г: {price_change:+.1f}%\n"
        
        if pump_type:
            report_text += f"🚨 Тип події: {pump_type}\n"
            report_text += f"⚠️ Рівень ризику: {pump_data.get('risk_level', 5)}/10\n"
        
        report_text += f"📊 Волатильність: {calculate_volatility(closes[-24:]):.1f}%\n"
        report_text += f"💹 Співвідношення обсягу: {volume_metrics.get('volume_ratio', 0):.1f}x\n"
        
        if is_volume_anomaly:
            report_text += "🔴 Виявлено аномалію обсягу!\n"
        
        # Додаємо рекомендацію
        if pump_type == "PUMP" and pump_data.get('risk_level', 5) > 7:
            report_text += "\n🔻 Рекомендація: Високий ризик! Уникайте входу.\n"
        elif pump_type == "PUMP":
            report_text += "\n🟡 Рекомендація: Обережно! Можлива корекція.\n"
        elif price_change < -10:
            report_text += "\n🟢 Рекомендація: Можливий відскок після падіння.\n"
        else:
            report_text += "\n⚪ Рекомендація: Стандартна ситуація.\n"
        
        bot.edit_message_text(report_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in advanced_analysis: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

# Додаємо відсутні команди
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    """Основна функція пошуку сигналів"""
    try:
        msg = bot.send_message(message.chat.id, "🔍 Аналізую ринок...")
        
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

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]
                last_price = closes[-1]

                # Технічні індикатори
                rsi, vol_spike = calculate_technical_indicators(closes, volumes)
                
                # Рівні підтримки/опору
                sr_levels = find_support_resistance(
                    closes, 
                    window=USER_SETTINGS['window_size'], 
                    delta=USER_SETTINGS['sensitivity']
                )

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100

                    if last_price > lvl * 1.01 and diff_pct > 1:
                        signal = (
                            f"🚀 LONG breakout\n"
                            f"Пробито опір: ${lvl:.4f}\n"
                            f"Поточна ціна: ${last_price:.4f}\n"
                            f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                        )
                        break
                    elif last_price < lvl * 0.99 and diff_pct < -1:
                        signal = (
                            f"⚡ SHORT breakout\n"
                            f"Пробито підтримку: ${lvl:.4f}\n"
                            f"Поточна ціна: ${last_price:.4f}\n"
                            f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                        )
                        break

                # Перевірка на памп/дамп
                event_type, price_change = detect_pump_dump(closes, volumes)
                
                if event_type:
                    signal = (
                        f"🔴 {event_type} DETECTED!\n"
                        f"Зміна ціни: {price_change:+.1f}%\n"
                        f"Рекомендація: {'Шорт' if event_type == 'PUMP' else 'Лонг'}\n"
                        f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                    )

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}\n" + "-"*40)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not signals:
            bot.edit_message_text("ℹ️ Жодних сигналів не знайдено.", message.chat.id, msg.message_id)
        else:
            text = f"<b>📊 Smart Auto Signals</b>\n\n" + "\n".join(signals[:10])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")

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

@bot.message_handler(commands=['settings'])
def show_settings(message):
    """Налаштування параметрів"""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
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

# Обробники для кнопок налаштувань
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
    
    @bot.message_handler(commands=['arbitrage'])
def arbitrage_handler(message):
    """Пошук арбітражних можливостей"""
    try:
        msg = bot.send_message(message.chat.id, "🔍 Шукаю арбітражні можливості...")
        
        # Отримуємо ціни
        prices = arbitrage_analyzer.get_ticker_prices()
        if not prices:
            bot.edit_message_text("❌ Не вдалося отримати дані з Binance", message.chat.id, msg.message_id)
            return
        
        # Шукаємо трикутні арбітражі
        opportunities = arbitrage_analyzer.find_triangular_arbitrage_pairs(prices)
        
        if not opportunities:
            bot.edit_message_text("ℹ️ Арбітражних можливостей не знайдено.", message.chat.id, msg.message_id)
            return
        
        # Формуємо повідомлення з топ-5 можливостей
        message_text = "<b>🔎 Знайдені арбітражні можливості:</b>\n\n"
        
        for i, opportunity in enumerate(opportunities[:5]):
            message_text += f"{i+1}. {arbitrage_analyzer.format_opportunity_message(opportunity)}\n"
            message_text += "─" * 40 + "\n"
        
        bot.edit_message_text(message_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in arbitrage: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['market_depth'])
def market_depth_handler(message):
    """Аналіз глибини ринку для арбітражу"""
    try:
        # Перевіряємо, чи вказано токен
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /market_depth BTCUSDT")
            return
            
        symbol = parts[1].upper()
        msg = bot.send_message(message.chat.id, f"🔍 Аналізую глибину ринку для {symbol}...")
        
        # Аналізуємо глибину ринку
        depth_analysis = arbitrage_analyzer.calculate_depth_arbitrage(symbol)
        
        if not depth_analysis:
            bot.edit_message_text("❌ Не вдалося проаналізувати глибину ринку", message.chat.id, msg.message_id)
            return
        
        # Формуємо звіт
        report_text = f"<b>📊 Аналіз глибини ринку {symbol}</b>\n\n"
        report_text += f"Найкраща ціна купівлі: {depth_analysis['best_bid']:.8f}\n"
        report_text += f"Найкраща ціна продажу: {depth_analysis['best_ask']:.8f}\n"
        report_text += f"Спред: {depth_analysis['spread']:.8f}\n"
        report_text += f"Спред (%): {depth_analysis['spread_percentage']:.4f}%\n"
        report_text += f"Обсяг купівлі (топ-5): {depth_analysis['bid_volume']:.4f}\n"
        report_text += f"Обсяг продажу (топ-5): {depth_analysis['ask_volume']:.4f}\n"
        report_text += f"Диспропорція: {depth_analysis['imbalance']:.4f}\n\n"
        
        # Додаємо рекомендацію
        if depth_analysis['spread_percentage'] < 0.1:
            report_text += "🟢 Низький спред - хороша ліквідність\n"
        elif depth_analysis['spread_percentage'] < 0.5:
            report_text += "🟡 Середній спред - помірна ліквідність\n"
        else:
            report_text += "🔴 Високий спред - низька ліквідність\n"
            
        if depth_analysis['imbalance'] > 2:
            report_text += "📈 Сильний дисбаланс у бік купівлі\n"
        elif depth_analysis['imbalance'] < 0.5:
            report_text += "📉 Сильний дисбаланс у бік продажу\n"
        else:
            report_text += "⚖️ Збалансований ринок\n"
        
        bot.edit_message_text(report_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in market_depth: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

if __name__ == "__main__":
    # Видаляємо вебхук якщо він був встановлений раніше
    bot.remove_webhook()
    
    # Запускаємо бота в режимі polling в окремому потоці
    def run_bot():
        logger.info("Запуск бота в режимі polling...")
        while True:
            try:
                bot.polling(none_stop=True, interval=3, timeout=20)
            except Exception as e:
                logger.error(f"Помилка бота: {e}")
                logger.info("Перезапуск бота через 10 секунд...")
                time.sleep(10)
    
    # Запускаємо бота в окремому потоці
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Запускаємо Flask сервер для Render
    port = int(os.environ.get('PORT', 5000))
    
    @app.route('/health')
    def health():
        return "OK"
    
    # Запускаємо Flask
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)