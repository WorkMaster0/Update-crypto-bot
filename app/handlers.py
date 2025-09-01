import requests
from telebot import types
import re
from datetime import datetime
import numpy as np
from app.bot import bot
from app.analytics import (
    get_price, get_klines, generate_signal_text, trend_strength_text,
    find_levels, top_movers, position_size, normalize_symbol, find_atr_squeeze
)
from app.chart import plot_candles
from app.config import DEFAULT_INTERVAL, ALLOWED_INTERVALS

# просте зберігання налаштувань чату в ОЗП
_user_defaults = {}  # chat_id -> {"interval": "1h"}

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

# ---------- /start ----------
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, (
        "🚀 <b>Crypto Analysis Bot</b> запущено!\n\n"
        "📊 <b>Основні команди:</b>\n"
        "• <code>/price BTCUSDT</code> - поточна ціна\n"
        "• <code>/chart BTCUSDT 1h</code> - графік з аналізом\n"
        "• <code>/levels BTCUSDT 4h</code> - рівні підтримки/опору\n"
        "• <code>/risk 1000 1 65000 64000</code> - розрахунок ризику\n\n"
        "🔍 <b>Аналіз:</b>\n"
        "• <code>/analyze BTCUSDT</code> - повний аналіз\n"
        "• <code>/scan gainers</code> - топ росту\n"
        "• <code>/scan volume</code> - топ обсягів\n"
        "• <code>/pattern BTCUSDT</code> - торгові паттерни\n\n"
        "🤖 <b>AI функції:</b>\n"
        "• <code>/ai_signal BTCUSDT</code> - AI сигнал\n"
        "• <code>/ai_scan</code> - AI сканування\n"
        "• <code>/ai_strategy BTCUSDT</code> - стратегія\n\n"
        "⚙️ <code>/setdefault 1h</code> - інтервал за замовчуванням\n"
        "📖 <code>/help</code> - довідка по командам"
    ), parse_mode="HTML")

# ---------- /help ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📊 Основні", callback_data="help_basic"),
        types.InlineKeyboardButton("🔍 Аналіз", callback_data="help_analysis"),
        types.InlineKeyboardButton("🤖 AI", callback_data="help_ai"),
        types.InlineKeyboardButton("⚙️ Налаштування", callback_data="help_settings")
    )
    
    bot.reply_to(message, "📖 <b>Оберіть категорію для довідки:</b>", 
                parse_mode="HTML", reply_markup=markup)

# ---------- /price ----------
@bot.message_handler(commands=['price'])
def price_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/price BTCUSDT</code>")
    try:
        price = get_price(symbol)
        bot.reply_to(message, f"💰 <b>{symbol}</b> = <b>{price:.6f}</b> USDT")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /chart ----------
@bot.message_handler(commands=['chart'])
def chart_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/chart BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        img = plot_candles(symbol, interval=interval, limit=200, with_levels=True)
        bot.send_photo(message.chat.id, img, caption=f"📊 <b>{symbol} [{interval}]</b>", parse_mode="HTML")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /levels ----------
@bot.message_handler(commands=['levels'])
def levels_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/levels BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        lv = find_levels(candles)
        s = ", ".join(f"{x:.4f}" for x in lv["supports"])
        r = ", ".join(f"{x:.4f}" for x in lv["resistances"])
        bot.reply_to(message, (
            f"🔎 <b>{symbol}</b> [{interval}] Levels\n"
            f"Supports: {s or '—'}\n"
            f"Resistances: {r or '—'}\n"
            f"Nearest S: <b>{lv['near_support']:.4f}</b> | "
            f"Nearest R: <b>{lv['near_resistance']:.4f}</b>\n"
            f"ATR(14): {lv['atr']:.4f}"
        ), parse_mode="HTML")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /analyze (ОБ'ЄДНАНА КОМАНДА) ----------
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/analyze BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🔍 Аналізую {symbol} [{interval}]...")
        
        # Отримуємо всі дані за один раз
        candles = get_klines(symbol, interval=interval)
        
        # Генеруємо сигнал
        signal_text = generate_signal_text(symbol, interval=interval)
        
        # Знаходимо рівні
        levels = find_levels(candles)
        s = ", ".join(f"{x:.4f}" for x in levels["supports"][-3:]) if levels["supports"] else "—"
        r = ", ".join(f"{x:.4f}" for x in levels["resistances"][-3:]) if levels["resistances"] else "—"
        
        # Аналізуємо тренд
        trend_text = trend_strength_text(candles)
        
        # Формуємо повну відповідь
        response = [
            f"🎯 <b>Повний аналіз {symbol} [{interval}]:</b>\n",
            f"📊 {signal_text}",
            f"",
            f"📈 <b>Тренд:</b> {trend_text}",
            f"",
            f"🔍 <b>Ключові рівні:</b>",
            f"• Підтримки: {s}",
            f"• Опори: {r}",
            f"• Найближча підтримка: <b>{levels['near_support']:.4f}</b>",
            f"• Найближчий опір: <b>{levels['near_resistance']:.4f}</b>",
            f"",
            f"📉 <b>Волатильність (ATR):</b> {levels['atr']:.4f}"
        ]
        
        # Додаємо кнопки для додаткових дій
        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("📊 Графік", callback_data=f"chart_{symbol}_{interval}"),
            types.InlineKeyboardButton("🔄 Оновити", callback_data=f"analyze_{symbol}_{interval}")
        )
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /scan (УНІФІКОВАНЕ СКАНУВАННЯ) ----------
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
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтруємо USDT пари
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 10000000 and
                float(d['lastPrice']) > 0.01)
        ]
        
        # Сортуємо за обраним критерієм
        key, reverse = scan_types[scan_type][1], scan_types[scan_type][2]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: abs(float(x[key])) if scan_type == 'breakouts' else float(x[key]), reverse=reverse)
        
        # Беремо топ-15
        top_pairs = sorted_pairs[:15]
        
        response = [f"📊 <b>{scan_types[scan_type][0]}:</b>\n"]
        
        for i, pair in enumerate(top_pairs, 1):
            symbol = pair['symbol']
            price = float(pair['lastPrice'])
            change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000  # В мільйонах
            
            emoji = "🟢" if change > 0 else "🔴"
            
            if scan_type in ['gainers', 'losers', 'breakouts']:
                response.append(f"{i}. {emoji} {symbol}: {change:+.2f}% | ${price:.4f}")
            else:
                response.append(f"{i}. {symbol}: ${price:.4f} | Vol: {volume:.1f}M")
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /risk ----------
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
        ), parse_mode="HTML")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /pattern ----------
@bot.message_handler(commands=['pattern'])
def pattern_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/pattern BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    
    try:
        processing_msg = bot.send_message(message.chat.id, f"🔍 Шукаю паттерни для {symbol} [{interval}]...")
        
        # Отримуємо дані
        candles = get_klines(symbol, interval=interval, limit=100)
        if not candles or len(candles['c']) < 20:
            bot.reply_to(message, f"❌ Недостатньо даних для {symbol} [{interval}]")
            return
        
        # Аналіз паттернів (спрощена версія)
        patterns = detect_patterns(candles)
        
        if not patterns:
            bot.reply_to(message, f"🔍 Для {symbol} [{interval}] паттернів не знайдено")
            return
        
        response = [f"🎯 <b>Знайдені паттерни для {symbol} [{interval}]:</b>\n"]
        
        for pattern_name, pattern_type, confidence in patterns[:5]:  # Обмежуємо 5 паттернами
            emoji = "🟢" if pattern_type == "BULLISH" else "🔴"
            response.append(f"{emoji} {pattern_name} ({confidence}% впевненості)")
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- AI КОМАНДИ (СПРОЩЕНІ) ----------
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
        
        # Додаємо AI рекомендацію
        recommendation = generate_ai_recommendation(symbol, interval)
        
        response = [
            f"🤖 <b>AI Сигнал для {symbol} [{interval}]:</b>\n",
            f"📊 {signal_text}",
            f"",
            f"💡 <b>AI Рекомендація:</b>",
            f"{recommendation}"
        ]
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=['ai_scan'])
def ai_scan_handler(message):
    try:
        processing_msg = bot.send_message(message.chat.id, "🤖 AI сканує ринок...")
        
        # Спрощене AI сканування
        opportunities = find_ai_opportunities()
        
        if not opportunities:
            bot.reply_to(message, "🔍 AI не знайшов сильних можливостей")
            return
        
        response = ["🤖 <b>AI Топ можливості:</b>\n"]
        
        for symbol, signal, confidence in opportunities[:5]:
            emoji = "🟢" if "LONG" in signal else "🔴"
            response.append(f"{emoji} {symbol}: {signal} ({confidence}% впевненості)")
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- ДОПОМІЖНІ ФУНКЦІЇ ----------
def detect_patterns(candles):
    """Спрощена детекція паттернів"""
    patterns = []
    # Тут буде логіка виявлення паттернів
    return patterns

def generate_ai_recommendation(symbol, interval):
    """Генерація AI рекомендації"""
    # Спрощена AI логіка
    return "Рекомендація буде тут на основі технічного аналізу"

def find_ai_opportunities():
    """Пошук AI можливостей"""
    # Спрощена логіка пошуку
    return []

# ---------- Callback обробники ----------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    try:
        if call.data.startswith('help_'):
            category = call.data.replace('help_', '')
            show_help_category(call, category)
        elif call.data.startswith('chart_'):
            data = call.data.split('_')
            symbol, interval = data[1], data[2]
            show_chart(call, symbol, interval)
        elif call.data.startswith('analyze_'):
            data = call.data.split('_')
            symbol, interval = data[1], data[2]
            reanalyze(call, symbol, interval)
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def show_help_category(call, category):
    help_texts = {
        'basic': "📊 <b>Основні команди:</b>\n• /price SYMBOL\n• /chart SYMBOL\n• /levels SYMBOL\n• /risk баланс риск% вход стоп",
        'analysis': "🔍 <b>Аналіз:</b>\n• /analyze SYMBOL\n• /scan TYPE\n• /pattern SYMBOL",
        'ai': "🤖 <b>AI:</b>\n• /ai_signal SYMBOL\n• /ai_scan\n• /ai_strategy SYMBOL",
        'settings': "⚙️ <b>Налаштування:</b>\n• /setdefault INTERVAL"
    }
    
    bot.edit_message_text(help_texts.get(category, "❌ Невідома категорія"),
                         call.message.chat.id, call.message.message_id,
                         parse_mode="HTML")

def show_chart(call, symbol, interval):
    try:
        img = plot_candles(symbol, interval=interval, limit=100)
        bot.send_photo(call.message.chat.id, img, caption=f"📊 {symbol} [{interval}]")
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

def reanalyze(call, symbol, interval):
    try:
        # Тут логіка повторного аналізу
        bot.answer_callback_query(call.id, "🔄 Оновлюю аналіз...")
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {e}")

# ---------- /setdefault ----------
@bot.message_handler(commands=['setdefault'])
def setdefault_handler(message):
    parts = message.text.split()
    if len(parts) < 2 or parts[1] not in ALLOWED_INTERVALS:
        return bot.reply_to(message, "⚠️ Приклад: <code>/setdefault 1h</code>")
    _user_defaults.setdefault(message.chat.id, {})["interval"] = parts[1]
    bot.reply_to(message, f"✅ Інтервал за замовчуванням для цього чату: <b>{parts[1]}</b>", parse_mode="HTML")