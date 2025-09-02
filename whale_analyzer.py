# whale_analyzer.py
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from utils import get_klines_cached, safe_request

logger = logging.getLogger(__name__)

class AdvancedWhaleAnalyzer:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.whale_threshold = 500000  # $500k для визначення кита
        
    def get_large_trades(self, symbol: str = "BTCUSDT", limit: int = 100) -> List[Dict]:
        """Отримати великі угоди з останніх даних"""
        try:
            url = f"{self.base_url}/trades"
            params = {'symbol': symbol, 'limit': limit}
            response = safe_request(url, params=params, timeout=10)
            
            if not response or not isinstance(response, list):
                return []
                
            large_trades = []
            for trade in response:
                if isinstance(trade, dict) and 'price' in trade and 'qty' in trade:
                    try:
                        trade_value = float(trade['price']) * float(trade['qty'])
                        if trade_value >= self.whale_threshold:
                            large_trades.append({
                                'symbol': symbol,
                                'price': float(trade['price']),
                                'quantity': float(trade['qty']),
                                'value': trade_value,
                                'time': datetime.fromtimestamp(trade['time']/1000) if 'time' in trade else datetime.now(),
                                'is_buyer': trade.get('isBuyerMaker', False)
                            })
                    except (ValueError, TypeError):
                        continue
            
            return large_trades
        except Exception as e:
            logger.error(f"Error getting large trades for {symbol}: {e}")
            return []
    
    def detect_whale_accumulation(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Виявити накопичення китами"""
        try:
            large_trades = self.get_large_trades(symbol, 500)
            
            if not large_trades:
                return None
            
            buy_volume = sum(trade['value'] for trade in large_trades if trade.get('is_buyer', False))
            sell_volume = sum(trade['value'] for trade in large_trades if not trade.get('is_buyer', True))
            
            if buy_volume > sell_volume * 3 and sell_volume > 0:
                return {
                    'symbol': symbol,
                    'type': 'ACCUMULATION',
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'ratio': buy_volume / sell_volume,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting whale accumulation for {symbol}: {e}")
            return None
    
    def detect_pump_preparation(self, symbol: str) -> Optional[Dict]:
        """Виявити підготовку до пампу"""
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=50"
            response = safe_request(url, timeout=10)
            
            if not response or not isinstance(response, dict):
                return None
                
            ask_orders = response.get('asks', [])[:20]
            large_ask_orders = []
            
            for order in ask_orders:
                if len(order) >= 2:
                    try:
                        price, quantity = float(order[0]), float(order[1])
                        order_value = price * quantity
                        if order_value > self.whale_threshold:
                            large_ask_orders.append({
                                'price': price,
                                'quantity': quantity,
                                'value': order_value
                            })
                    except (ValueError, TypeError):
                        continue
            
            if large_ask_orders:
                total_value = sum(order['value'] for order in large_ask_orders)
                return {
                    'symbol': symbol,
                    'type': 'PUMP_PREPARATION',
                    'large_orders_count': len(large_ask_orders),
                    'total_value': total_value,
                    'orders': large_ask_orders[:5]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pump preparation for {symbol}: {e}")
            return None
    
    def detect_dump_warning(self, symbol: str) -> Optional[Dict]:
        """Виявити ознаки майбутнього дампу"""
        try:
            large_trades = self.get_large_trades(symbol, 200)
            
            if not large_trades:
                return None
            
            recent_sells = [t for t in large_trades if not t.get('is_buyer', True)]
            recent_buys = [t for t in large_trades if t.get('is_buyer', False)]
            
            sell_volume = sum(t['value'] for t in recent_sells)
            buy_volume = sum(t['value'] for t in recent_buys)
            
            if sell_volume > buy_volume * 2 and len(recent_sells) > 5 and buy_volume > 0:
                return {
                    'symbol': symbol,
                    'type': 'DUMP_WARNING',
                    'sell_volume': sell_volume,
                    'buy_volume': buy_volume,
                    'sell_count': len(recent_sells),
                    'buy_count': len(recent_buys),
                    'ratio': sell_volume / buy_volume
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting dump warning for {symbol}: {e}")
            return None
    
    def analyze_token_whale_activity(self, symbol: str) -> Optional[Dict]:
    """Детальний аналіз китової активності для токена"""
    try:
        # Пропускаємо стейблкоїни
        stablecoins = ['USDC', 'FDUSD', 'BUSD', 'TUSD', 'USDP', 'DAI', 'PAX']
        if any(stablecoin in symbol for stablecoin in stablecoins):
            return None
            
        # Решта коду залишається без змін...
        large_trades = self.get_large_trades(symbol, 200)
        if not large_trades:
            return None
            
        two_hours_ago = datetime.now() - timedelta(hours=2)
        recent_trades = [t for t in large_trades if t['time'] > two_hours_ago]
        
        if not recent_trades:
            return None
        
        # ВИПРАВЛЕНІ ВІДСТУПИ - ці рядки мають бути на одному рівні
        buy_volume = sum(t['value'] for t in recent_trades if t.get('is_buyer', False))
        sell_volume = sum(t['value'] for t in recent_trades if not t.get('is_buyer', True))
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return None
            
        buy_ratio = buy_volume / total_volume
        
        klines = get_klines_cached(symbol, interval="15m", limit=20)
        if not klines or not klines.get('c'):
            return None
        
        closes = klines['c']
        if len(closes) < 2:
            return None
            
        price_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
        
        activity_type = "NEUTRAL"
        if buy_ratio > 0.7 and price_change > 2:
            activity_type = "STRONG_BUYING"
        elif buy_ratio < 0.3 and price_change < -2:
            activity_type = "STRONG_SELLING"
        elif buy_ratio > 0.6:
            activity_type = "BUYING"
        elif buy_ratio < 0.4:
            activity_type = "SELLING"
        
        return {
            'symbol': symbol,
            'activity_type': activity_type,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_ratio': buy_ratio,
            'price_change': price_change,
            'trade_count': len(recent_trades),
            'total_volume': total_volume
        }
            
    except Exception as e:
        logger.error(f"Error in detailed analysis for {symbol}: {e}")
        return None
    
    def get_high_volume_symbols(self, min_volume: float = 10000000) -> List[str]:
    """Отримати токени з високим обсягом торгів (без стейблкоїнів)"""
    try:
        url = f"{self.base_url}/ticker/24hr"
        data = safe_request(url, timeout=15)
        
        if not data or not isinstance(data, list):
            return []
        
        # Стейблкоїни, які потрібно виключити
        stablecoins = ['USDC', 'FDUSD', 'BUSD', 'TUSD', 'USDP', 'DAI', 'PAX']
            
        high_volume_symbols = [
            d for d in data 
            if isinstance(d, dict) and 
            d.get('symbol', '').endswith('USDT') and 
            float(d.get('quoteVolume', 0)) > min_volume and
            not any(stablecoin in d.get('symbol', '') for stablecoin in stablecoins)
        ]
        
        high_volume_symbols.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
        return [s['symbol'] for s in high_volume_symbols[:30]]
        
    except Exception as e:
        logger.error(f"Error getting high volume symbols: {e}")
        return []

# Глобальний екземпляр аналізатора
whale_analyzer = AdvancedWhaleAnalyzer()