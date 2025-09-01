# app/analytics/indicators.py
import numpy as np
from typing import Tuple
from app.analytics.data import get_klines
from app.config import DEFAULT_INTERVAL

# ---------- EMA ----------
def ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    series = np.asarray(series, dtype=float)
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

# ---------- ATR ----------
def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (returns ATR series using EMA smoothing)."""
    # True range
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    return ema(tr, period)

# ---------- RSI ----------
def calculate_rsi(close_prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI computed via Wilder's smoothing (EMA used here)."""
    close_prices = np.asarray(close_prices, dtype=float)
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # pad to align length (first element NaN)
    if len(close_prices) < 2:
        return np.array([np.nan] * len(close_prices))

    avg_gain = ema(np.concatenate(([gain[0]], gain)), period)
    avg_loss = ema(np.concatenate(([loss[0]], loss)), period)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.concatenate(([np.nan], rsi))  # shift to match input length
    return rsi

# ---------- MACD ----------
def calculate_macd(close_prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(close_prices, fast)
    ema_slow = ema(close_prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ---------- MTF Trend helper ----------
def get_multi_timeframe_trend(symbol: str, current_interval: str = DEFAULT_INTERVAL) -> str:
    """
    Simple multi-timeframe trend: check higher TF EMA50 vs EMA200.
    Returns: "STRONG_UP", "STRONG_DOWN", or "NEUTRAL"
    """
    higher_tf_map = {
        '1m': '5m', '5m': '15m', '15m': '30m',
        '30m': '1h', '1h': '4h', '4h': '1d', '1d': '1w'
    }
    higher_interval = higher_tf_map.get(current_interval, '4h')
    try:
        candles = get_klines(symbol, interval=higher_interval, limit=250)
        c = candles["c"]
        e50 = ema(c, 50)
        e200 = ema(c, 200)
        if e50[-1] > e200[-1] * 1.02:
            return "STRONG_UP"
        elif e50[-1] < e200[-1] * 0.98:
            return "STRONG_DOWN"
        else:
            return "NEUTRAL"
    except Exception:
        return "NEUTRAL"