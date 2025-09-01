from telebot import TeleBot, types
import requests
import numpy as np
import re
from datetime import datetime
from app.config import TELEGRAM_BOT_TOKEN
from app.analytics import (
    get_price, get_klines, generate_signal_text, trend_strength_text,
    find_levels, top_movers, position_size, normalize_symbol,
    find_atr_squeeze, detect_liquidity_trap, calculate_rsi, calculate_macd,
    get_multi_timeframe_trend, get_crypto_sentiment
)
from app.chart import plot_candles
from app.config import DEFAULT_INTERVAL, ALLOWED_INTERVALS

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

bot = TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

# Глобальні змінні для налаштувань
_user_defaults = {}
_notify_settings = {}

def _default_interval(chat_id):
    return _user_defaults.get(chat_id, {}).get("interval", DEFAULT_INTERVAL)

def _parse_args(msg_text: str):
    parts = msg_text.split()
    symbol = None
    interval = None
    if len(parts) >= 2:
        symbol = normalize_symbol(parts[1])
    if len(parts) >= 3 and parts[2] in ALLOWED_INTERVALS:
        interval = parts[2]
    return symbol, interval

# ---------- КОМАНДИ ВІДПАЛЮВАННЯ ----------
@bot.message_handler(commands=['start', 'help'])
def start_handler(message):
    """Головне меню з інтерактивними кнопками"""
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        types.KeyboardButton("📊 Аналіз монети"),
        types.KeyboardButton("🔍 Сканування ринку"),
        types.KeyboardButton("🤖 AI сигнали"),
        types.KeyboardButton("⚙️ Налаштування")
    )
    
    welcome_text = """
🚀 <b>Crypto Analysis Pro Bot</b>

<b>Основні команди:</b>
• /analyze BTCUSDT - повний аналіз
• /scan gainers - топ росту
• /scan volume - топ обсягів
• /risk 1000 1 65000 64000 - розрахунок ризику

<b>AI функції:</b>
• /ai_signal BTCUSDT - AI сигнал
• /ai_strategy BTCUSDT - стратегія
• /ai_scan - AI сканування

<b>Додатково:</b>
• /chart BTCUSDT - графік
• /levels BTCUSDT - рівні
• /price BTCUSDT - ціна
• /setdefault 1h - інтервал

🎯 <i>Використовуйте кнопки нижче для швидкого доступу</i>
    """
    
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)

# ---------- ОБРОБНИКИ КНОПОК ----------
@bot.message_handler(func=lambda message: message.text == "📊 Аналіз монети")
def analyze_button_handler(message):
    bot.send_message(message.chat.id, "🔍 Введіть назву монети для аналізу (наприклад: BTCUSDT):")
    bot.register_next_step_handler(message, process_analyze_symbol)

def process_analyze_symbol(message):
    symbol = normalize_symbol(message.text)
    analyze_handler(message, symbol)

@bot.message_handler(func=lambda message: message.text == "🔍 Сканування ринку")
def scan_button_handler(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📈 Топ росту", callback_data="scan_gainers"),
        types.InlineKeyboardButton("📉 Топ падіння", callback_data="scan_losers"),
        types.InlineKeyboardButton("💎 Високий обсяг", callback_data="scan_volume"),
        types.InlineKeyboardButton("🚀 Пробої", callback_data="scan_breakouts")
    )
    bot.send_message(message.chat.id, "🔍 Оберіть тип сканування:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "🤖 AI сигнали")
def ai_button_handler(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📊 AI сигнал", callback_data="ai_signal_menu"),
        types.InlineKeyboardButton("🔍 AI сканування", callback_data="ai_scan"),
        types.InlineKeyboardButton("🎯 AI стратегія", callback_data="ai_strategy_menu")
    )
    bot.send_message(message.chat.id, "🤖 Оберіть AI функцію:", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "⚙️ Налаштування")
def settings_button_handler(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    intervals = ['1h', '4h', '1d', '15m', '1m']
    buttons = [types.InlineKeyboardButton(f"⏰ {iv}", callback_data=f"set_interval_{iv}") for iv in intervals]
    markup.add(*buttons)
    bot.send_message(message.chat.id, "⚙️ Оберіть інтервал за замовчуванням:", reply_markup=markup)

# ---------- ОСНОВНІ КОМАНДИ ----------
@bot.message_handler(commands=['analyze'])
def analyze_command_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        bot.send_message(message.chat.id, "🔍 Введіть назву монети для аналізу (наприклад: /analyze BTCUSDT):")
        bot.register_next_step_handler(message, process_analyze_command)
    else:
        analyze_handler(message, symbol, interval)

def process_analyze_command(message):
    symbol = normalize_symbol(message.text)
    analyze_handler(message, symbol)

def analyze_handler(message, symbol, interval=None):
    """Уніфікований аналіз монети"""
    interval = interval or _default_interval(message.chat.id)
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🔍 Аналізую {symbol} [{interval}]...")
        
        # Отримуємо всі дані
        candles = get_klines(symbol, interval=interval)
        signal_text = generate_signal_text(symbol, interval=interval)
        levels = find_levels(candles)
        trend_text = trend_strength_text(candles)
        
        # Додаткові метрики
        rsi = calculate_rsi(candles["c"], 14)[-1]
        macd_line, signal_line, macd_hist = calculate_macd(candles["c"])
        htf_trend = get_multi_timeframe_trend(symbol, interval)
        
        # Формуємо повну відповідь
        response = [
            f"🎯 <b>Повний аналіз {symbol} [{interval}]</b>",
            f"",
            f"📊 {signal_text.split('|')[0]}",  # Перший рядок сигналу
            f"",
            f"📈 <b>Технічні показники:</b>",
            f"• RSI(14): {rsi:.1f}",
            f"• MACD Hist: {macd_hist[-1]:.4f}",
            f"• Тренд старшого TF: {htf_trend}",
            f"",
            f"🔍 <b>Ключові рівні:</b>",
            f"• Підтримка: {levels['near_support'] or 'N/A':.4f}",
            f"• Опір: {levels['near_resistance'] or 'N/A':.4f}",
            f"• ATR: {levels['atr']:.4f}",
            f"",
            f"💡 <b>Рекомендація:</b> {_generate_recommendation(signal_text, rsi, macd_hist[-1])}"
        ]
        
        # Кнопки дій
        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(
            types.InlineKeyboardButton("📊 Графік", callback_data=f"chart_{symbol}_{interval}"),
            types.InlineKeyboardButton("🔄 Оновити", callback_data=f"reanalyze_{symbol}_{interval}"),
            types.InlineKeyboardButton("🤖 Детальний AI аналіз", callback_data=f"ai_full_{symbol}")
        )
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка аналізу: {str(e)}")

def _generate_recommendation(signal_text, rsi, macd_hist):
    """Генерація рекомендації на основі технічних показників"""
    if "LONG" in signal_text and rsi < 70 and macd_hist > 0:
        return "🟢 Можливий лонг - чекайте підтвердження"
    elif "SHORT" in signal_text and rsi > 30 and macd_hist < 0:
        return "🔴 Можливий шорт - чекайте підтвердження"
    elif abs(rsi - 50) < 10:
        return "🟡 Нейтрально - ринок у консолідації"
    else:
        return "⚪️ Чекайте чітких сигналів"

@bot.message_handler(commands=['scan'])
def scan_handler(message):
    parts = message.text.split()
    scan_type = 'gainers' if len(parts) < 2 else parts[1].lower()
    
    scan_types = {
        'gainers': ('Топ росту', 'priceChangePercent', True),
        'losers': ('Топ падіння', 'priceChangePercent', False),
        'volume': ('Топ обсягів', 'quoteVolume', True),
        'breakouts': ('Пробої', 'priceChangePercent', True),
        'liquid': ('Ліквідність', 'quoteVolume', True)
    }
    
    if scan_type not in scan_types:
        return bot.reply_to(message, "⚠️ Доступні типи: gainers, losers, volume, breakouts, liquid")
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🔍 Сканую {scan_types[scan_type][0]}...")
        
        # Отримуємо дані
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтруємо та сортуємо
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 10000000 and
                float(d['lastPrice']) > 0.01)
        ]
        
        key, reverse = scan_types[scan_type][1], scan_types[scan_type][2]
        sorted_pairs = sorted(usdt_pairs, 
                            key=lambda x: abs(float(x[key])) if scan_type == 'breakouts' else float(x[key]), 
                            reverse=reverse)
        
        # Формуємо відповідь
        response = [f"📊 <b>{scan_types[scan_type][0]}:</b>\n"]
        
        for i, pair in enumerate(sorted_pairs[:15], 1):
            symbol = pair['symbol']
            price = float(pair['lastPrice'])
            change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000
            
            emoji = "🟢" if change > 0 else "🔴"
            
            if scan_type in ['gainers', 'losers', 'breakouts']:
                response.append(f"{i}. {emoji} {symbol}: {change:+.2f}% | ${price:.4f}")
            else:
                response.append(f"{i}. {symbol}: ${price:.4f} | Vol: {volume:.1f}M")
        
        # Додаємо кнопки для швидкого аналізу
        markup = types.InlineKeyboardMarkup()
        for pair in sorted_pairs[:3]:
            markup.add(types.InlineKeyboardButton(
                f"📊 {pair['symbol']}", 
                callback_data=f"analyze_{pair['symbol']}"
            ))
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка сканування: {str(e)}")

@bot.message_handler(commands=['price'])
def price_handler(message):
    symbol, _ = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/price BTCUSDT</code>")
    try:
        price = get_price(symbol)
        bot.reply_to(message, f"💰 <b>{symbol}</b> = <b>{price:.6f}</b> USDT")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=['chart'])
def chart_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/chart BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        img = plot_candles(symbol, interval=interval, limit=200, with_levels=True)
        bot.send_photo(message.chat.id, img, caption=f"📊 <b>{symbol} [{interval}]</b>")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=['levels'])
def levels_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/levels BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        lv = find_levels(candles)
        s = ", ".join(f"{x:.4f}" for x in lv["supports"][-3:])
        r = ", ".join(f"{x:.4f}" for x in lv["resistances"][-3:])
        bot.reply_to(message, (
            f"🔎 <b>{symbol}</b> [{interval}] Levels\n"
            f"Supports: {s or '—'}\n"
            f"Resistances: {r or '—'}\n"
            f"Nearest S: <b>{lv['near_support']:.4f}</b> | "
            f"Nearest R: <b>{lv['near_resistance']:.4f}</b>\n"
            f"ATR(14): {lv['atr']:.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=['risk'])
def risk_handler(message):
    parts = message.text.split()
    if len(parts) < 5:
        return bot.reply_to(message, "⚠️ Приклад: <code>/risk 1000 1 65000 64000</code> (balance risk% entry stop)")
    try:
        balance = float(parts[1])
        risk_pct = float(parts[2])
        entry = float(parts[3])
        stop = float(parts[4])
        res = position_size(balance, risk_pct, entry, stop)
        bot.reply_to(message, (
            f"🧮 Risk: {risk_pct:.2f}% від ${balance:.2f} → ${res['risk_amount']:.2f}\n"
            f"📦 Position size ≈ <b>{res['qty']:.6f}</b> токенів\n"
            f"🎯 1R ≈ {abs(entry - stop):.4f} | 2R TP ≈ {entry + (res['rr_one_tp'] if entry>stop else -res['rr_one_tp']):.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- AI КОМАНДИ ----------
@bot.message_handler(commands=['ai_signal'])
def ai_signal_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/ai_signal BTCUSDT</code>")
    interval = interval or _default_interval(message.chat.id)
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🤖 AI аналізує {symbol}...")
        
        # Отримуємо сигнал
        signal_text = generate_signal_text(symbol, interval=interval)
        
        # Додатковий AI аналіз
        sentiment_value, sentiment_text = get_crypto_sentiment()
        htf_trend = get_multi_timeframe_trend(symbol, interval)
        
        response = [
            f"🤖 <b>AI Сигнал для {symbol} [{interval}]:</b>\n",
            f"📊 {signal_text}",
            f"",
            f"🎭 <b>Настрої ринку:</b> {sentiment_value or 'N/A'} ({sentiment_text})",
            f"📈 <b>Старший TF:</b> {htf_trend}",
            f"",
            f"💡 <b>AI рекомендація:</b>",
            f"{_generate_ai_recommendation(signal_text, sentiment_value, htf_trend)}"
        ]
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response))
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

def _generate_ai_recommendation(signal_text, sentiment_value, htf_trend):
    """Генерація AI рекомендації"""
    if "LONG" in signal_text and sentiment_value and sentiment_value < 40 and "UP" in htf_trend:
        return "🟢 СИЛЬНИЙ лонг сигнал - входити на відкатах"
    elif "SHORT" in signal_text and sentiment_value and sentiment_value > 60 and "DOWN" in htf_trend:
        return "🔴 СИЛЬНИЙ шорт сигнал - входити на відскоках"
    elif "LONG" in signal_text:
        return "🟢 Лонг сигнал - чекайте підтвердження"
    elif "SHORT" in signal_text:
        return "🔴 Шорт сигнал - чекайте підтвердження"
    else:
        return "⚪️ Немає чітких сигналів - чекайте"

@bot.message_handler(commands=['ai_scan'])
def ai_scan_handler(message):
    try:
        processing_msg = bot.send_message(message.chat.id, "🤖 AI сканує ринок...")
        
        # Отримуємо топ монети
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 50000000 and
                float(d['lastPrice']) > 0.01)
        ]
        
        # Скануємо кожну монету
        opportunities = []
        for pair in usdt_pairs[:20]:  # Обмежуємо для швидкості
            symbol = pair['symbol']
            try:
                signal_text = generate_signal_text(symbol, interval="1h")
                if any(keyword in signal_text for keyword in ['LONG', 'SHORT', 'BUY', 'SELL']):
                    change = float(pair['priceChangePercent'])
                    volume = float(pair['quoteVolume']) / 1000000
                    opportunities.append((symbol, signal_text, change, volume))
            except:
                continue
        
        if not opportunities:
            bot.reply_to(message, "🔍 AI не знайшов сильних можливостей")
            return
        
        # Сортуємо за обсягом
        opportunities.sort(key=lambda x: x[3], reverse=True)
        
        response = ["🤖 <b>AI Топ можливості:</b>\n"]
        
        for symbol, signal, change, volume in opportunities[:8]:
            direction = "🟢" if "LONG" in signal or "BUY" in signal else "🔴"
            response.append(f"{direction} <b>{symbol}</b> - {change:+.2f}% | Vol: {volume:.1f}M")
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response))
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=['ai_strategy'])
def ai_strategy_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/ai_strategy BTCUSDT</code>")
    interval = interval or _default_interval(message.chat.id)
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🎯 AI створює стратегію для {symbol}...")
        
        # Отримуємо дані
        candles = get_klines(symbol, interval=interval)
        levels = find_levels(candles)
        current_price = candles["c"][-1]
        
        # Генеруємо стратегію
        strategy = _generate_trading_strategy(symbol, candles, levels, current_price)
        
        response = [
            f"🎯 <b>AI Стратегія для {symbol} [{interval}]:</b>\n",
            f"📊 Поточна ціна: ${current_price:.4f}",
            f"",
            f"🔍 <b>Ключові рівні:</b>",
            f"• Підтримка: {levels['near_support'] or 'N/A':.4f}",
            f"• Опір: {levels['near_resistance'] or 'N/A':.4f}",
            f"",
            f"🚀 <b>Стратегія:</b>",
            f"{strategy}",
            f"",
            f"💡 <b>Ризик-менеджмент:</b>",
            f"• Ризик на угоду: 1-2%",
            f"• Stop Loss: 1-2x ATR ({levels['atr']:.4f})",
            f"• Take Profit: 2-3x Risk"
        ]
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response))
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

def _generate_trading_strategy(symbol, candles, levels, current_price):
    """Генерація торгової стратегії"""
    rsi = calculate_rsi(candles["c"], 14)[-1]
    macd_line, signal_line, macd_hist = calculate_macd(candles["c"])
    
    if rsi < 35 and macd_hist[-1] > 0 and levels['near_support']:
        return f"🟢 ЛОНГ стратегія: Вхід біля {levels['near_support']:.4f}, SL: {levels['near_support'] - levels['atr']:.4f}"
    elif rsi > 65 and macd_hist[-1] < 0 and levels['near_resistance']:
        return f"🔴 ШОРТ стратегія: Вхід біля {levels['near_resistance']:.4f}, SL: {levels['near_resistance'] + levels['atr']:.4f}"
    else:
        return "⚪️ Консолідація: Чекайте пробою рівнів або відскоку"

# ---------- CALLBACK HANDLERS ----------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    try:
        if call.data.startswith('chart_'):
            data = call.data.split('_')
            symbol, interval = data[1], data[2]
            show_chart(call, symbol, interval)
        elif call.data.startswith('analyze_'):
            symbol = call.data.replace('analyze_', '')
            analyze_callback(call, symbol)
        elif call.data.startswith('reanalyze_'):
            data = call.data.split('_')
            symbol, interval = data[1], data[2]
            reanalyze(call, symbol, interval)
        elif call.data.startswith('scan_'):
            scan_type = call.data.replace('scan_', '')
            scan_callback(call, scan_type)
        elif call.data.startswith('set_interval_'):
            interval = call.data.replace('set_interval_', '')
            set_interval_callback(call, interval)
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def show_chart(call, symbol, interval):
    try:
        img = plot_candles(symbol, interval=interval, limit=100)
        bot.send_photo(call.message.chat.id, img, caption=f"📊 {symbol} [{interval}]")
        bot.answer_callback_query(call.id, "📊 Графік завантажено")
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def analyze_callback(call, symbol):
    try:
        bot.answer_callback_query(call.id, f"🔍 Аналізую {symbol}...")
        analyze_handler(call.message, symbol)
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def reanalyze(call, symbol, interval):
    try:
        bot.answer_callback_query(call.id, "🔄 Оновлюю аналіз...")
        analyze_handler(call.message, symbol, interval)
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def scan_callback(call, scan_type):
    try:
        fake_msg = type('obj', (object,), {'chat': type('obj', (object,), {'id': call.message.chat.id})})
        scan_handler(fake_msg, scan_type)
        bot.answer_callback_query(call.id, "🔍 Сканування завершено")
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def set_interval_callback(call, interval):
    try:
        _user_defaults.setdefault(call.message.chat.id, {})["interval"] = interval
        bot.answer_callback_query(call.id, f"✅ Інтервал встановлено: {interval}")
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

# ---------- SET DEFAULT INTERVAL ----------
@bot.message_handler(commands=['setdefault'])
def setdefault_handler(message):
    parts = message.text.split()
    if len(parts) < 2 or parts[1] not in ALLOWED_INTERVALS:
        return bot.reply_to(message, "⚠️ Приклад: <code>/setdefault 1h</code>")
    _user_defaults.setdefault(message.chat.id, {})["interval"] = parts[1]
    bot.reply_to(message, f"✅ Інтервал за замовчуванням: <b>{parts[1]}</b>")

# ---------- Запуск бота ----------
if __name__ == "__main__":
    print("🚀 Бот запущено...")
    bot.infinity_polling()
