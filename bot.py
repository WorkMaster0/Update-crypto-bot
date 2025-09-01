import os
import requests
import logging
from datetime import datetime
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

# Налаштування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("BOT_TOKEN не знайдено в змінних оточення")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# Глобальні змінні для налаштувань
USER_SETTINGS = {
    'min_volume': 5000000,
    'top_symbols': 30,
    'window_size': 20,
    'sensitivity': 0.005
}

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

# Команди бота (залишаються без змін, як у вашому попередньому коді)
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Привітальне повідомлення"""
    help_text = """
🤖 Smart Crypto Bot - Аналіз пампів та дампів

Доступні команды:
/smart_auto - Автоматичний пошук сигналів
/settings - Налаштування параметрів
/check_token <token> - Перевірити конкретний токен
/stats - Статистика ринку
"""
    bot.reply_to(message, help_text)

@bot.message_handler(commands=['settings'])
def show_settings(message):
    """Налаштування параметрів"""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("Мін. обсяг 📊"))
    keyboard.add(KeyboardButton("Кількість монет 🔢"))
    keyboard.add(KeyboardButton("Чутливість ⚖️"))
    keyboard.add(KeyboardButton("Головне меню 🏠"))
    
    settings_text = f"""
Поточні налаштування:

Мінімальний обсяг: {USER_SETTINGS['min_volume']:,.0f} USDT
Кількість монет для аналізу: {USER_SETTINGS['top_symbols']}
Чутливість: {USER_SETTINGS['sensitivity'] * 100}%
"""
    bot.send_message(message.chat.id, settings_text, reply_markup=keyboard)

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
                if len(closes) >= 24:
                    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
                else:
                    price_change_24h = 0
                
                if abs(price_change_24h) > 15 and vol_spike:
                    direction = "PUMP" if price_change_24h > 0 else "DUMP"
                    signal = (
                        f"🔴 {direction} DETECTED!\n"
                        f"Зміна ціни: {price_change_24h:+.1f}%\n"
                        f"Обсяг: {'екстремальний' if vol_spike else 'підвищений'}\n"
                        f"RSI: {rsi:.1f}"
                    )

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}\n" + "-"*40)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not signals:
            bot.edit_message_text("ℹ️ Жодних сигналів не знайдено.", message.chat.id, msg.message_id)
        else:
            text = f"<b>📊 Smart Auto Signals</b>\n\n" + "\n".join(signals[:10])  # Обмежуємо кількість сигналів
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
        
        analysis_text = f"""
<b>{symbol} Analysis</b>

Поточна ціна: ${last_price:.4f}
RSI: {rsi:.1f} {'(перекупленість)' if rsi > 70 else '(перепроданість)' if rsi < 30 else ''}
Обсяг: {'підвищений' if vol_spike else 'нормальний'}

<b>Key Levels:</b>
"""
        for level in sr_levels[-5:]:  # Останні 5 рівнів
            distance_pct = (last_price - level) / level * 100
            analysis_text += f"{level:.4f} ({distance_pct:+.1f}%)\n"

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
    logger.info("Бот запущений...")
    bot.polling(none_stop=True)
