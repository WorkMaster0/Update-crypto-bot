# utils.py (додаємо цю функцію)
def get_klines_cached(symbol: str, interval: str = "1h", limit: int = 100) -> Optional[Dict]:
    """Отримання клінів з кешуванням"""
    cache_key = f"klines_{symbol}_{interval}_{limit}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    
    try:
        url = f"https://api.binance.com/api/v3/klines"
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