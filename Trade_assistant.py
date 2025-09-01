import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TradeAssistant:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Отримати комплексні дані по токену"""
        try:
            # Отримуємо різні види даних
            klines = self.get_klines(symbol, "1h", 100)
            ticker = self.get_ticker_24hr(symbol)
            depth = self.get_depth(symbol)
            
            if not all([klines, ticker, depth]):
                return None
                
            return {
                'klines': klines,
                'ticker': ticker,
                'depth': depth,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def generate_trade_signal(self, symbol: str) -> Dict:
        """Згенерувати торговий сигнал для токена"""
        market_data = self.get_market_data(symbol)
        if not market_data:
            return {'error': 'Could not fetch market data'}
        
        # Аналізуємо різні аспекти
        trend_analysis = self.analyze_trend(market_data['klines'])
        volume_analysis = self.analyze_volume(market_data['klines'])
        momentum_analysis = self.analyze_momentum(market_data['klines'])
        liquidity_analysis = self.analyze_liquidity(market_data['depth'])
        
        # Генеруємо рекомендацію
        recommendation = self.generate_recommendation(
            trend_analysis,
            volume_analysis, 
            momentum_analysis,
            liquidity_analysis
        )
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': self.calculate_confidence(
                trend_analysis,
                volume_analysis,
                momentum_analysis
            ),
            'entry_points': self.calculate_entry_points(market_data['klines']),
            'exit_points': self.calculate_exit_points(market_data['klines']),
            'risk_level': self.calculate_risk_level(market_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_trend(self, klines: List) -> Dict:
        """Аналіз тренду"""
        closes = [float(k[4]) for k in klines]
        price_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
        
        return {
            'direction': 'up' if price_change > 0 else 'down',
            'strength': abs(price_change),
            'trend_type': self.determine_trend_type(closes)
        }
    
    def analyze_volume(self, klines: List) -> Dict:
        """Аналіз обсягів"""
        volumes = [float(k[5]) for k in klines]
        current_volume = volumes[-1] if volumes else 0
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else current_volume
        
        return {
            'current_volume': current_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'volume_trend': 'increasing' if current_volume > avg_volume else 'decreasing'
        }
    
    def analyze_momentum(self, klines: List) -> Dict:
        """Аналіз моментуму"""
        closes = [float(k[4]) for k in klines]
        rsi = self.calculate_rsi(closes)
        
        return {
            'rsi': rsi,
            'momentum': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
            'price_acceleration': self.calculate_acceleration(closes)
        }
    
    def analyze_liquidity(self, depth: Dict) -> Dict:
        """Аналіз ліквідності"""
        bids = depth.get('bids', [])[:5]
        asks = depth.get('asks', [])[:5]
        
        bid_volume = sum(float(bid[1]) for bid in bids) if bids else 0
        ask_volume = sum(float(ask[1]) for ask in asks) if asks else 0
        
        return {
            'bid_liquidity': bid_volume,
            'ask_liquidity': ask_volume,
            'spread_percentage': self.calculate_spread_percentage(bids, asks),
            'order_book_imbalance': self.calculate_imbalance(bids, asks)
        }
    
    def generate_recommendation(self, trend: Dict, volume: Dict, momentum: Dict, liquidity: Dict) -> str:
        """Генерація торгової рекомендації"""
        if momentum['rsi'] > 70 and trend['strength'] > 10:
            return "STRONG_SELL"
        elif momentum['rsi'] < 30 and trend['strength'] > 10:
            return "STRONG_BUY"
        elif volume['volume_ratio'] > 2 and trend['direction'] == 'up':
            return "BUY"
        elif volume['volume_ratio'] > 2 and trend['direction'] == 'down':
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_confidence(self, trend: Dict, volume: Dict, momentum: Dict) -> float:
        """Розрахунок впевненості в сигналі"""
        confidence = 50
        
        if trend['strength'] > 20:
            confidence += 20
        elif trend['strength'] > 10:
            confidence += 10
            
        if volume['volume_ratio'] > 2:
            confidence += 15
            
        if momentum['rsi'] > 70 or momentum['rsi'] < 30:
            confidence += 15
            
        return min(95, max(5, confidence))
    
    def calculate_entry_points(self, klines: List) -> List[float]:
        """Розрахунок точок входу"""
        closes = [float(k[4]) for k in klines]
        current_price = closes[-1] if closes else 0
        
        return [
            current_price * 0.98,
            current_price * 0.95, 
            current_price * 0.92
        ]
    
    def calculate_exit_points(self, klines: List) -> List[float]:
        """Розрахунок точок виходу"""
        closes = [float(k[4]) for k in klines]
        current_price = closes[-1] if closes else 0
        
        return [
            current_price * 1.05,
            current_price * 1.08,
            current_price * 1.12
        ]
    
    def calculate_risk_level(self, market_data: Dict) -> str:
        """Розрахунок рівня ризику"""
        volatility = self.calculate_volatility([float(k[4]) for k in market_data['klines']])
        
        if volatility > 10:
            return "HIGH"
        elif volatility > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Допоміжні функції
    def get_klines(self, symbol: str, interval: str, limit: int) -> Optional[List]:
        try:
            url = f"{self.base_url}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except:
            return None
    
    def get_ticker_24hr(self, symbol: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=10)
            return response.json()
        except:
            return None
    
    def get_depth(self, symbol: str) -> Optional[Dict]:
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            response = requests.get(url, timeout=10)
            return response.json()
        except:
            return None
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0
            
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return sum(abs(r) for r in returns) / len(returns) * 100
    
    def determine_trend_type(self, prices: List[float]) -> str:
        if len(prices) < 10:
            return "short_term"
            
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        
        if short_ma > long_ma * 1.05:
            return "strong_uptrend"
        elif short_ma > long_ma:
            return "weak_uptrend"
        elif short_ma < long_ma * 0.95:
            return "strong_downtrend"
        else:
            return "weak_downtrend"
    
    def calculate_acceleration(self, prices: List[float]) -> float:
        if len(prices) < 3:
            return 0
            
        recent_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        previous_change = (prices[-2] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
        
        return (recent_change - previous_change) * 100
    
    def calculate_spread_percentage(self, bids: List, asks: List) -> float:
        if not bids or not asks:
            return 0
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        return ((best_ask - best_bid) / best_bid) * 100 if best_bid != 0 else 0
    
    def calculate_imbalance(self, bids: List, asks: List) -> float:
        if not bids or not asks:
            return 1
            
        bid_volume = sum(float(bid[1]) for bid in bids[:3])
        ask_volume = sum(float(ask[1]) for ask in asks[:3])
        
        return bid_volume / ask_volume if ask_volume > 0 else float('inf')

# Глобальний екземпляр
trade_assistant = TradeAssistant()