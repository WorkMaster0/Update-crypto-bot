# app/analytics/signals.py
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.analytics.data import get_klines
from app.analytics.indicators import ema, atr, calculate_rsi, calculate_macd, get_multi_timeframe_trend
from app.analytics.levels import find_levels

# ---------- Helpers ----------
def _last(vals: np.ndarray, default=float("nan")):
    return float(vals[-1]) if len(vals) else default

def compute_atr_squeeze(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14, ma_period: int = 20) -> float:
    """Повертає ratio current_atr / mean(last ma_period ATR). Значення <1 означає стиснення."""
    atr_series = atr(h, l, c, period)
    if len(atr_series) < ma_period:
        return 1.0
    current_atr = float(atr_series[-1])
    ma = float(np.mean(atr_series[-ma_period:]))
    if ma == 0:
        return 1.0
    return current_atr / ma

# ---------- Basic pattern detectors ----------
def detect_engulfing(o: np.ndarray, c: np.ndarray) -> Optional[Tuple[str, str]]:
    """
    Простий детектор engulfing (останні 2 свічки).
    Повертає (pattern_name, direction) або None.
    """
    if len(c) < 2:
        return None
    prev_o, prev_c = float(o[-2]), float(c[-2])
    cur_o, cur_c = float(o[-1]), float(c[-1])

    # Bullish engulfing
    if prev_c < prev_o and cur_c > cur_o and cur_c > prev_o and cur_o < prev_c:
        return ("BULLISH_ENGULFING", "LONG")
    # Bearish engulfing
    if prev_c > prev_o and cur_c < cur_o and cur_c < prev_o and cur_o > prev_c:
        return ("BEARISH_ENGULFING", "SHORT")
    return None

def detect_flag_like(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[Tuple[str,str]]:
    """
    Проста heuristic перевірка прапора (flag): сильний рух за 20 свічок + коротка консолідація 5 свічок.
    Повертає ("BULL_FLAG"/"BEAR_FLAG", direction) або None.
    """
    if len(closes) < 30:
        return None
    price_change = (closes[-1] - closes[-20]) / closes[-20]
    if abs(price_change) < 0.05:  # мінімум 5% рух
        return None
    last5_range = float(np.max(highs[-5:]) - np.min(lows[-5:]))
    prev5_range = float(np.max(highs[-10:-5]) - np.min(lows[-10:-5]) + 1e-9)
    if last5_range < prev5_range * 0.6:
        return ("BULL_FLAG", "LONG") if price_change > 0 else ("BEAR_FLAG", "SHORT")
    return None

# ---------- Core checks ----------
def check_trend(c: np.ndarray) -> Tuple[Optional[str], int]:
    """Повертає ('UP'/'DOWN'/'FLAT' or None, score:int)"""
    if len(c) < 200:
        # якщо мало даних — робимо м'якше правило
        e50 = ema(c, 50) if len(c) >= 50 else None
        e200 = ema(c, 200) if len(c) >= 200 else None
    else:
        e50 = ema(c, 50)
        e200 = ema(c, 200)

    if e50 is None or e200 is None:
        return (None, 0)

    last50, last200 = e50[-1], e200[-1]
    score = 0
    if last50 > last200 * 1.01:
        score = 2
        return ("UP", score)
    elif last50 < last200 * 0.99:
        score = 2
        return ("DOWN", score)
    else:
        return ("FLAT", 0)

def check_macd_signal(macd_hist: np.ndarray) -> Tuple[Optional[str], int]:
    """Оцінка по MACD histogram: positive -> bullish, negative -> bearish"""
    if len(macd_hist) < 1:
        return (None, 0)
    v = float(macd_hist[-1])
    if v > 0:
        return ("BULL", 1)
    if v < 0:
        return ("BEAR", 1)
    return (None, 0)

def check_rsi_signal(rsi: np.ndarray) -> Tuple[Optional[str], int]:
    """RSI conditions: oversold/overbought or neutral"""
    if len(rsi) < 1 or np.isnan(rsi[-1]):
        return (None, 0)
    val = float(rsi[-1])
    if val < 30:
        return ("OVERSOLD", 2)
    if val > 70:
        return ("OVERBOUGHT", 2)
    if 35 <= val <= 65:
        return ("NEUTRAL", 1)
    return (None, 0)

def check_volume_spike(vols: np.ndarray) -> Tuple[bool, int]:
    """Поверхневий check: останній об'єм > mean(last20) * 1.2"""
    if len(vols) < 5:
        return (False, 0)
    avg = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    last = float(vols[-1])
    if avg <= 0:
        return (False, 0)
    if last > avg * 1.5:
        return (True, 2)
    if last > avg * 1.2:
        return (True, 1)
    return (False, 0)

def check_levels_proximity(last_price: float, levels: Dict[str, Any], atr_val: float) -> Tuple[Optional[str], int, Dict[str, float]]:
    """
    Перевіряє відношення ціни до найближчих S/R і повертає напрямок/score і рекомендовані entry/stop/targets приблизно.
    """
    near_s = levels.get("near_support")
    near_r = levels.get("near_resistance")
    info = {}
    score = 0
    direction = None

    # Bounce from support -> LONG
    if near_s is not None and last_price > near_s and (last_price - near_s) <= max(atr_val, last_price * 0.004):
        direction = "LONG"
        score += 2
        entry = last_price
        stop = float(near_s - max(atr_val * 0.5, last_price * 0.0025))
        target = levels.get("near_resistance") or float(last_price + 2.0 * atr_val)
        info.update({"entry": round(entry, 6), "stop": round(stop, 6), "target": round(target, 6)})
        return (direction, score, info)

    # Rejection from resistance -> SHORT
    if near_r is not None and last_price < near_r and (near_r - last_price) <= max(atr_val, last_price * 0.004):
        direction = "SHORT"
        score += 2
        entry = last_price
        stop = float(near_r + max(atr_val * 0.5, last_price * 0.0025))
        target = levels.get("near_support") or float(last_price - 2.0 * atr_val)
        info.update({"entry": round(entry,6), "stop": round(stop,6), "target": round(target,6)})
        return (direction, score, info)

    # Breakout above resistance
    if near_r is not None and last_price > near_r * 1.01:
        direction = "LONG"
        score += 3
        entry = float(near_r * 1.001)  # small buffer
        stop = float(near_r - max(atr_val * 0.5, last_price * 0.0025))
        target = float(last_price + 2.0 * atr_val)
        info.update({"entry": round(entry,6), "stop": round(stop,6), "target": round(target,6)})
        return (direction, score, info)

    # Breakdown below support
    if near_s is not None and last_price < near_s * 0.99:
        direction = "SHORT"
        score += 3
        entry = float(near_s * 0.999)
        stop = float(near_s + max(atr_val * 0.5, last_price * 0.0025))
        target = float(last_price - 2.0 * atr_val)
        info.update({"entry": round(entry,6), "stop": round(stop,6), "target": round(target,6)})
        return (direction, score, info)

    return (None, 0, {})

# ---------- AI ALERT CORE ----------
def ai_alert_core(symbol: str, interval: str = "1h", klines_limit: int = 200) -> Dict[str, Any]:
    """
    Головна функція для AI alert (перший рівень).
    Повертає структуру з direction, confidence, entry/stop/targets, reasons (list).
    """
    # Забираємо дані
    candles = get_klines(symbol, interval=interval, limit=klines_limit)
    o, h, l, c, v = candles["o"], candles["h"], candles["l"], candles["c"], candles["v"]
    last = float(c[-1])
    reasons: List[str] = []
    total_score = 0

    # Indicators
    try:
        atr_val = float(atr(h, l, c, 14)[-1])
    except Exception:
        atr_val = float(np.std(c[-14:])) if len(c) >= 14 else max(0.0, last * 0.01)

    rsi = calculate_rsi(c, 14)
    macd_line, signal_line, macd_hist = calculate_macd(c)
    mtf_trend = get_multi_timeframe_trend(symbol, interval)

    # Levels
    levels = find_levels(candles)

    # 1) trend check
    tf_direction, tf_score = check_trend(c)
    if tf_direction == "UP":
        total_score += tf_score
        reasons.append("EMA50 > EMA200 (Up)")
    elif tf_direction == "DOWN":
        total_score -= tf_score
        reasons.append("EMA50 < EMA200 (Down)")

    # 2) macd
    macd_dir, macd_score = check_macd_signal(macd_hist)
    if macd_dir == "BULL":
        total_score += macd_score
        reasons.append("MACD positive")
    elif macd_dir == "BEAR":
        total_score -= macd_score
        reasons.append("MACD negative")

    # 3) rsi
    rsi_dir, rsi_score = check_rsi_signal(rsi)
    if rsi_dir == "OVERSOLD":
        total_score += rsi_score
        reasons.append(f"RSI {rsi[-1]:.1f} (oversold)")
    elif rsi_dir == "OVERBOUGHT":
        total_score -= rsi_score
        reasons.append(f"RSI {rsi[-1]:.1f} (overbought)")
    elif rsi_dir == "NEUTRAL":
        total_score += rsi_score * 0  # small neutral effect

    # 4) volume
    vol_spike, vol_score = check_volume_spike(v)
    if vol_spike:
        total_score += vol_score
        reasons.append("Volume spike")

    # 5) levels proximity / breakout
    lvl_dir, lvl_score, lvl_info = check_levels_proximity(last, levels, atr_val)
    if lvl_dir:
        # lvl_score adds positive or negative depending on direction
        total_score += lvl_score if lvl_dir == "LONG" else -lvl_score
        reasons.append(f"Level signal: {lvl_dir}")

    # 6) ATR squeeze
    squeeze_ratio = compute_atr_squeeze(h, l, c)
    if squeeze_ratio < 0.75:
        total_score += 1
        reasons.append(f"ATR squeeze ({squeeze_ratio:.2f})")

    # 7) multi-timeframe
    if mtf_trend == "STRONG_UP":
        total_score += 2
        reasons.append("Higher TF strong up")
    elif mtf_trend == "STRONG_DOWN":
        total_score -= 2
        reasons.append("Higher TF strong down")

    # 8) simple pattern checks
    pat = detect_engulfing(o, c)
    if pat:
        pname, pdirection = pat
        if pdirection == "LONG":
            total_score += 2
            reasons.append("Bullish engulfing")
        else:
            total_score -= 2
            reasons.append("Bearish engulfing")

    flag = detect_flag_like(h, l, c)
    if flag:
        pname, pdirection = flag
        total_score += 1 if pdirection == "LONG" else -1
        reasons.append(pname)

    # Aggregate direction
    direction = None
    if total_score >= 4:
        direction = "LONG"
    elif total_score <= -4:
        direction = "SHORT"
    else:
        direction = None

    # Confidence mapping (simple)
    confidence = max(20, min(95, 50 + int(total_score * 6)))  # base 50, +/- per score

    # Build result
    result = {
        "symbol": symbol,
        "interval": interval,
        "last_price": round(last, 8),
        "direction": direction,
        "confidence": confidence,
        "score": total_score,
        "reasons": reasons,
        "levels": levels,
        "squeeze_ratio": round(float(squeeze_ratio), 3)
    }

    # Attach suggested entry/stop/targets
    if lvl_info:
        result.update(lvl_info)
        # add nicer TP list
        entry = lvl_info.get("entry", last)
        # targets: nearest level or ATR-based
        t1 = lvl_info.get("target", entry + 2 * atr_val if direction == "LONG" else entry - 2 * atr_val)
        t2 = entry + 4 * atr_val if direction == "LONG" else entry - 4 * atr_val
        result["targets"] = [round(t1, 6), round(t2, 6)]
    else:
        # fallback entries
        if direction == "LONG":
            entry = last
            stop = round(last - max(atr_val * 1.0, last * 0.01), 6)
            result.update({"entry": round(entry,6), "stop": stop, "targets": [round(entry + 2*atr_val,6), round(entry + 4*atr_val,6)]})
        elif direction == "SHORT":
            entry = last
            stop = round(last + max(atr_val * 1.0, last * 0.01), 6)
            result.update({"entry": round(entry,6), "stop": stop, "targets": [round(entry - 2*atr_val,6), round(entry - 4*atr_val,6)]})
        else:
            # no signal
            result.update({"entry": None, "stop": None, "targets": []})

    return result

# ---------- Text formatter (helper) ----------
def format_alert_text(alert: Dict[str, Any]) -> str:
    """
    Форматований текст для відправки в Telegram.
    """
    lines: List[str] = []
    sym = alert.get("symbol", "")
    interval = alert.get("interval", "")
    last = alert.get("last_price")
    direction = alert.get("direction")
    conf = alert.get("confidence", 0)

    lines.append(f"🎯 <b>AI Alert: {sym} [{interval}]</b>")
    lines.append(f"Last: {last:.6f} | Confidence: {conf}%")
    if direction:
        lines.append(f"Direction: <b>{direction}</b>")
        if alert.get("entry"):
            lines.append(f"Entry: {alert['entry']:.6f} | SL: {alert['stop']:.6f}")
        if alert.get("targets"):
            t = ", ".join(f"{x:.6f}" for x in alert["targets"])
            lines.append(f"Targets: {t}")
    else:
        lines.append("ℹ️ No clear actionable signal — wait for confirmation.")

    if alert.get("reasons"):
        lines.append("\nReasons:")
        for r in alert["reasons"][:6]:
            lines.append(f"• {r}")

    # include squeeze ratio & nearest levels summary
    lines.append(f"\nSqueeze ratio: {alert.get('squeeze_ratio')}")
    lv = alert.get("levels", {})
    s = ", ".join(f"{x:.6f}" for x in (lv.get("supports") or [])[:4])
    r = ", ".join(f"{x:.6f}" for x in (lv.get("resistances") or [])[:4])
    lines.append(f"Levels S: [{s or '—'}] | R: [{r or '—'}]")

    return "\n".join(lines)