# app/analytics/indicators.py
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
from cachetools import cached, TTLCache
from app.config import BINANCE_BASES, HTTP_TIMEOUT, KLINES_LIMIT, DEFAULT_INTERVAL, PIVOT_LEFT_RIGHT, MAX_LEVELS

# ---------- BINANCE HELPERS (CACHED) ----------
def _binance_get(path: str, params: Dict) -> dict:
    last_error = None
    for base in BINANCE_BASES:
        try:
            r = requests.get(f"{base}{path}", params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("Binance unreachable")

def normalize_symbol(s: str) -> str:
    return s.strip().upper().replace("/", "")

@cached(cache=TTLCache(maxsize=100, ttl=60))
def get_price(symbol: str) -> float:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

def get_klines(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = KLINES_LIMIT) -> Dict[str, np.ndarray]:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    arr = np.array(data, dtype=object)
    ts = (arr[:,0].astype(np.int64) // 1000).astype(np.int64)
    o = arr[:,1].astype(float)
    h = arr[:,2].astype(float)
    l = arr[:,3].astype(float)
    c = arr[:,4].astype(float)
    v = arr[:,5].astype(float)
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}

# ---------- TECHNICAL INDICATORS ----------
def ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return ema(tr, period)

def calculate_rsi(close_prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.concatenate(([np.nan], rsi))
    return rsi

def calculate_macd(close_prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = ema(close_prices, fast)
    ema_slow = ema(close_prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ---------- SUPPORT/RESISTANCE LEVELS ----------
def _pivot_high(h: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(h), i + left_right + 1)
    return h[i] == np.max(h[left:right])

def _pivot_low(l: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(l), i + left_right + 1)
    return l[i] == np.min(l[left:right])

def find_levels(candles: Dict[str, np.ndarray], left_right: int = PIVOT_LEFT_RIGHT, max_levels: int = MAX_LEVELS) -> Dict[str, Optional[List[float]]]:
    h, l, c = candles["h"], candles["l"], candles["c"]
    last_price = c[-1]
    _atr = atr(h, l, c, 14)[-1]
    tol = max(_atr * 0.5, last_price * 0.002)
    highs, lows = [], []

    for i in range(left_right, len(h) - left_right):
        if _pivot_high(h, i, left_right):
            highs.append(h[i])
        if _pivot_low(l, i, left_right):
            lows.append(l[i])

    def cluster(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: List[List[float]] = [[levels[0]]]
        for x in levels[1:]:
            if abs(x - np.mean(clusters[-1])) <= tol:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        weighted = [(float(np.mean(g)), len(g)) for g in clusters]
        weighted.sort(key=lambda e: (-e[1], abs(e[0] - last_price)))
        return [w[0] for w in weighted[:max_levels]]

    resistances = cluster(highs)
    supports = cluster(lows)
    near_support = max([s for s in supports if s <= last_price], default=None)
    near_resist = min([r for r in resistances if r >= last_price], default=None)

    return {
        "supports": supports,
        "resistances": resistances,
        "near_support": near_support,
        "near_resistance": near_resist,
        "atr": float(_atr),
        "tolerance": float(tol),
        "last_price": float(last_price),
    }

# ---------- ATR SQUEEZE ----------
def find_atr_squeeze(symbol: str, interval: str = '1h', limit: int = 100) -> float:
    try:
        candles = get_klines(symbol, interval=interval, limit=limit)
        h, l, c = candles["h"], candles["l"], candles["c"]
        atr_values = atr(h, l, c, 14)
        if len(atr_values) < 20:
            return 1.0
        return float(atr_values[-1]) / float(np.mean(atr_values[-20:]))
    except Exception as e:
        print(f"ATR squeeze error for {symbol}: {e}")
        return 1.0

# ---------- LIQUIDITY TRAPS ----------
def detect_liquidity_trap(symbol: str, interval: str = "1h", lookback: int = 50) -> Optional[str]:
    candles = get_klines(symbol, interval=interval, limit=lookback)
    h, l, c, v = candles["h"], candles["l"], candles["c"], candles["v"]
    local_high = max(h[:-1])
    local_low = min(l[:-1])
    last_close = c[-1]
    last_high = h[-1]
    last_low = l[-1]
    last_vol = v[-1]
    avg_vol = np.mean(v[:-1]) if len(v) > 1 else last_vol

    trap_signal = None
    if last_high > local_high and last_close < local_high and last_vol > 1.5 * avg_vol:
        trap_signal = f"🐻 <b>Short Trap</b> on {symbol} – fake breakout up!"
    elif last_low < local_low and last_close > local_low and last_vol > 1.5 * avg_vol:
        trap_signal = f"🐂 <b>Long Trap</b> on {symbol} – fake breakout down!"
    return trap_signal