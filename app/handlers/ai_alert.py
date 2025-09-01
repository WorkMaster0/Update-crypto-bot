# handlers/ai_alert.py
from app.analytics import (
    get_klines, ema, atr, calculate_rsi, calculate_macd,
    find_levels, get_multi_timeframe_trend, get_crypto_sentiment,
    find_atr_squeeze, detect_liquidity_trap
)
from app.bot import bot
from datetime import datetime
import numpy as np

def generate_ai_signal(symbol: str, interval: str = "1h") -> str:
    """
    Генерує сигнал для користувача.
    Враховує:
    - Тренд по EMA50/200
    - RSI
    - MACD
    - ATR / squeeze
    - Multi-timeframe тренд
    - Патерни (auto pattern)
    - Sentiment (Fear & Greed)
    """
    candles = get_klines(symbol, interval=interval)
    c, h, l, v = candles["c"], candles["h"], candles["l"], candles["v"]
    last_price = c[-1]

    # --- Трендові індикатори ---
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    trend = "UP" if ema50[-1] > ema200[-1] else ("DOWN" if ema50[-1] < ema200[-1] else "FLAT")

    rsi = calculate_rsi(c)
    macd_line, signal_line, macd_hist = calculate_macd(c)
    atr_value = atr(h, l, c)[-1]

    # --- Рівні support/resistance ---
    levels = find_levels(candles)
    sup = levels["near_support"]
    res = levels["near_resistance"]

    # --- Multi-timeframe ---
    ht_trend = get_multi_timeframe_trend(symbol, interval)

    # --- Патерни ---
    pattern_signal = detect_liquidity_trap(symbol, interval)

    # --- ATR Squeeze ---
    squeeze_ratio = find_atr_squeeze(symbol, interval)
    squeeze_info = "ATR Squeeze!" if squeeze_ratio < 0.7 else None

    # --- Sentiment ---
    sentiment_value, sentiment_text = get_crypto_sentiment()

    # --- Формування тексту ---
    txt = [f"📊 <b>{symbol.upper()}</b> [{interval}] | Price: {last_price:.4f}"]
    txt.append(f"Trend: {trend} | EMA50: {ema50[-1]:.4f} | EMA200: {ema200[-1]:.4f}")
    txt.append(f"RSI(14): {rsi[-1]:.2f} | MACD Hist: {macd_hist[-1]:.4f}")
    txt.append(f"HTF Trend: {ht_trend}")
    if sentiment_value:
        txt.append(f"🎭 Fear & Greed: {sentiment_value} ({sentiment_text})")
    if pattern_signal:
        txt.append(pattern_signal)
    if squeeze_info:
        txt.append(squeeze_info)

    # --- Конфлюєнс ---
    confluence_score = 0
    reasons = []

    # LONG logic
    if sup and last_price > sup and (last_price - sup) <= max(atr_value, last_price*0.004):
        signal_dir = "LONG"
        entry = last_price
        stop = sup - atr_value*0.5
        tp = res if res else last_price + 2*atr_value

        if 30 < rsi[-1] < 70:
            confluence_score += 1
            reasons.append("RSI ok")
        if macd_hist[-1] > 0:
            confluence_score += 1
            reasons.append("MACD bullish")
        if trend == "UP":
            confluence_score += 1
            reasons.append("EMA trend UP")
        if ht_trend == "STRONG_UP":
            confluence_score += 2
            reasons.append("HTF UP")
        if sentiment_value and sentiment_value < 30:
            confluence_score += 1
            reasons.append("Extreme Fear")

    # SHORT logic
    elif res and last_price < res and (res - last_price) <= max(atr_value, last_price*0.004):
        signal_dir = "SHORT"
        entry = last_price
        stop = res + atr_value*0.5
        tp = sup if sup else last_price - 2*atr_value

        if 30 < rsi[-1] < 70:
            confluence_score += 1
            reasons.append("RSI ok")
        if macd_hist[-1] < 0:
            confluence_score += 1
            reasons.append("MACD bearish")
        if trend == "DOWN":
            confluence_score += 1
            reasons.append("EMA trend DOWN")
        if ht_trend == "STRONG_DOWN":
            confluence_score += 2
            reasons.append("HTF DOWN")
        if sentiment_value and sentiment_value > 70:
            confluence_score += 1
            reasons.append("Extreme Greed")
    else:
        signal_dir = None
        entry = stop = tp = None

    # --- Формування фінального повідомлення ---
    if signal_dir and confluence_score >= 3:
        txt.append(f"✅ <b>{signal_dir} CONFLUENCE ({confluence_score}/7)</b>")
        txt.append(f"Reason: {', '.join(reasons)}")
        txt.append(f"Entry ~{entry:.4f}, SL {stop:.4f}, TP {tp:.4f}")
    elif signal_dir:
        txt.append(f"🟡 Weak {signal_dir} signal ({confluence_score}/7). Reason: {', '.join(reasons)}")
    else:
        txt.append("ℹ️ No clear entry points. Wait for new candle or interval change.")

    return "\n".join(txt)