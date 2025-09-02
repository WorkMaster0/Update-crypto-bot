# utils.py
import time
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class Cache:
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, data):
        self.cache[key] = data

cache = Cache()

def safe_request(url, params=None, timeout=15):
    import requests
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    return None

def get_klines_cached(symbol: str, interval: str = "1h", limit: int = 100) -> Optional[Dict]:
    """Отримання клінів з кешуванням"""
    cache_key = f"klines_{symbol}_{interval}_{limit}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = safe_request(url, params=params)
        
        if data and isinstance(data, list):
            result = {
                'o': [float(c[1]) for c in data],
                'h': [float(c[2]) for c in data],
                'l': [float(c[3]) for c in data],
                'c': [float(c[4]) for c in data],
                'v': [float(c[5]) for c in data],
                't': [c[0] for c in data]
            }
            cache.set(cache_key, result)
            return result
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {e}")
    
    return None