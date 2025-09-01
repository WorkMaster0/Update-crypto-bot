import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_klines(self, symbol, interval="1h", limit=200):
        """Отримання історичних даних з Binance"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            data = requests.get(url, params=params, timeout=10).json()
            
            if not data or 'code' in data:
                return None
                
            df = {
                'o': [float(c[1]) for c in data],
                'h': [float(c[2]) for c in data],
                'l': [float(c[3]) for c in data],
                'c': [float(c[4]) for c in data],
                'v': [float(c[5]) for c in data],
                't': [c[0] for c in data]
            }
            return df
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return None
    
    def get_ticker_24hr(self, symbol=None):
        """Отримання 24годинної статистики"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            if symbol:
                url += f"?symbol={symbol}"
            
            data = requests.get(url, timeout=10).json()
            return data
        except Exception as e:
            logger.error(f"Error getting 24hr ticker: {e}")
            return None
    
    def get_exchange_info(self):
        """Отримання інформації про біржу"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            data = requests.get(url, timeout=10).json()
            return data
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return None