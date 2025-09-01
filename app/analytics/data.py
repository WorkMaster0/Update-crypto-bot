# app/analytics/data.py
import requests
import numpy as np
from typing import Dict
from cachetools import cached, TTLCache

from app.config import BINANCE_BASES, HTTP_TIMEOUT, KLINES_LIMIT, DEFAULT_INTERVAL

# --- HTTP helper with retries across BINANCE_BASES ---
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
    """Нормалізує символ до формату BINANCE (без '/', uppercase)."""
    return s.strip().upper().replace("/", "")

# Кешуємо ціну на 60 секунд
@cached(cache=TTLCache(maxsize=100, ttl=60))
def get_price(symbol: str) -> float:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

# Кеш для klines — короткий TTL, щоб не перевантажувати API при частих запитах
@cached(cache=TTLCache(maxsize=500, ttl=20))
def get_klines(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = KLINES_LIMIT) -> Dict[str, np.ndarray]:
    """
    Повертає OHLCV як numpy-масиви:
    {"t": timestamps_sec, "o": opens, "h": highs, "l": lows, "c": closes, "v": volumes}
    """
    symbol = normalize_symbol(symbol)
    limit = int(limit)
    data = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        raise RuntimeError(f"No klines for {symbol} {interval}")

    arr = np.array(data, dtype=object)
    # Binance returns milliseconds timestamps
    ts = (arr[:, 0].astype(np.int64) // 1000).astype(np.int64)
    o = arr[:, 1].astype(float)
    h = arr[:, 2].astype(float)
    l = arr[:, 3].astype(float)
    c = arr[:, 4].astype(float)
    v = arr[:, 5].astype(float)

    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}