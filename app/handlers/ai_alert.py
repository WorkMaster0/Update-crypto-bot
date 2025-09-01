# app/handlers/ai_alert.py
from typing import Optional, Dict
from app.analytics.indicators import (
    get_klines, ema, atr, calculate_rsi, calculate_macd,
    find_levels, get_multi_timeframe_trend, find_atr_squeeze,
    detect_liquidity_trap
)
from datetime import datetime
import numpy as np

def generate_ai_signal(symbol: str, interval: str = "1h") -> Dict:
    candles = get_klines(symbol, interval=interval)
    c, h, l, v = candles["c"], candles["h"], candles["l"], candles["v"]
    last_price = c[-1]

    # Основні індикатори
    e50 = ema(c, 50)
    e200 = ema(c, 200)
    rsi = calculate_rsi(c)
    macd_line, signal_line, macd_hist = calculate_macd(c)

    # Підтримка / Опір
    levels = find_levels(candles)
    sup = levels["near_support"]
    res = levels["near_resistance"]

    # Мультитаймфрейм тренд
    htf_trend = get_multi_timeframe_trend(symbol, interval)

    # ATR-сжаття
    squeeze_ratio = find_atr_squeeze(symbol, interval)

    # Пастки ліквідності
    trap_signal = detect_liquidity_trap(symbol, interval)

    # Патерн (простий приклад: подвійне дно / верх)
    pattern_signal = None
    if len(c) >= 20:
        if c[-2] < c[-3] and c[-1] > c[-2]:
            pattern_signal = "Double Bottom? ↑"
        elif c[-2] > c[-3] and c[-1] < c[-2]:
            pattern_signal = "Double Top? ↓"

    # Логіка сигналу
    confluence = 0
    reason = []
    direction = None

    # LONG
    if sup and last_price > sup and (last_price - sup) <= max(atr(h, l, c)[-1], last_price*0.004):
        direction = "LONG"
        if 30 < rsi[-1] < 70:
            confluence += 1; reason.append("RSI ok")
        if macd_hist[-1] > 0:
            confluence += 1; reason.append("MACD Bull")
        if htf_trend == "STRONG_UP":
            confluence += 2; reason.append("HTF UP")
        if squeeze_ratio < 0.75:
            confluence += 1; reason.append("Squeeze")
        if pattern_signal and "Bottom" in pattern_signal:
            confluence += 1; reason.append("Pattern Bottom")

    # SHORT
    elif res and last_price < res and (res - last_price) <= max(atr(h, l, c)[-1], last_price*0.004):
        direction = "SHORT"
        if 30 < rsi[-1] < 70:
            confluence += 1; reason.append("RSI ok")
        if macd_hist[-1] < 0:
            confluence += 1; reason.append("MACD Bear")
        if htf_trend == "STRONG_DOWN":
            confluence += 2; reason.append("HTF Down")
        if squeeze_ratio < 0.75:
            confluence += 1; reason.append("Squeeze")
        if pattern_signal and "Top" in pattern_signal:
            confluence += 1; reason.append("Pattern Top")

    signal_text = f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} | {symbol} | "
    if direction and confluence >= 3:
        signal_text += f"✅ {direction} CONFLUENCE ({confluence}/6) | Reasons: {', '.join(reason)}"
    elif direction:
        signal_text += f"🟡 Weak {direction} ({confluence}/6) | Reasons: {', '.join(reason)}"
    else:
        signal_text += "ℹ️ No clear signal"

    if trap_signal:
        signal_text += f"\n⚠️ Liquidity Trap: {trap_signal}"
    if pattern_signal:
        signal_text += f"\n🎯 Pattern: {pattern_signal}"

    return {
        "symbol": symbol,
        "interval": interval,
        "direction": direction,
        "confluence": confluence,
        "signal_text": signal_text,
        "levels": levels,
        "trap_signal": trap_signal,
        "pattern_signal": pattern_signal,
        "squeeze_ratio": squeeze_ratio,
        "htf_trend": htf_trend
    }