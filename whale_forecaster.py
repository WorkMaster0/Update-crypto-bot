# whale_forecaster.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import asyncio
from whale_analyzer import whale_analyzer
from utils import safe_request

logger = logging.getLogger(__name__)

class WhaleForecaster:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.orderbook_snapshots = {}
        self.whale_threshold = 50000  # $50k для кластерів
        
    async def analyze_orderbook_patterns(self, symbol: str) -> Optional[Dict]:
        """Асинхронний аналіз патернів у стакані"""
        try:
            depth = await asyncio.to_thread(safe_request, f"{self.base_url}/depth?symbol={symbol}&limit=50")
            if not depth or not isinstance(depth, dict):
                return None
            
            bids = depth.get('bids', [])[:20]  # Тільки топ-20
            asks = depth.get('asks', [])[:20]  # Тільки топ-20
            
            bid_clusters = self._find_order_clusters(bids, 'bid')
            ask_clusters = self._find_order_clusters(asks, 'ask')
            
            direction = self._determine_direction(bid_clusters, ask_clusters)
            
            if direction:
                return {
                    'symbol': symbol,
                    'expected_direction': direction,
                    'bid_clusters': len(bid_clusters),
                    'ask_clusters': len(ask_clusters),
                    'large_blocks': len(bid_clusters) + len(ask_clusters),
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Orderbook analysis failed for {symbol}: {e}")
            return None
    
    def _find_order_clusters(self, orders: List, order_type: str) -> List[Dict]:
        """Знаходження кластерів великих ордерів"""
        clusters = []
        
        for order in orders:
            if len(order) >= 2:
                try:
                    price = float(order[0])
                    quantity = float(order[1])
                    value = price * quantity
                    
                    if value >= self.whale_threshold:
                        clusters.append({
                            'price': price,
                            'quantity': quantity,
                            'value': value,
                            'type': order_type
                        })
                except (ValueError, TypeError):
                    continue
        
        return clusters
    
    def _determine_direction(self, bid_clusters: List, ask_clusters: List) -> Optional[str]:
        """Визначення потенційного напрямку"""
        if not bid_clusters and not ask_clusters:
            return None
        
        bid_pressure = sum(cluster['value'] for cluster in bid_clusters)
        ask_pressure = sum(cluster['value'] for cluster in ask_clusters)
        
        if bid_pressure > ask_pressure * 2:  # Більш консервативний поріг
            return 'BUY'
        elif ask_pressure > bid_pressure * 2:
            return 'SELL'
        
        return None
    
    async def predict_whale_movements(self) -> List[Dict]:
        """Швидке передбачення китових активностей"""
        try:
            # Отримуємо символи з високим обсягом
            high_volume_symbols = await asyncio.to_thread(whale_analyzer.get_high_volume_symbols)
            
            if not high_volume_symbols:
                return []
            
            # Аналізуємо тільки топ-5 символів для швидкості
            symbols_to_analyze = high_volume_symbols[:5]
            
            predictions = []
            
            # Асинхронно аналізуємо всі символи паралельно
            tasks = [self.analyze_orderbook_patterns(symbol) for symbol in symbols_to_analyze]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result and result['expected_direction']:
                    # Спрощена логіка впевненості
                    confidence = 60 + min(result['large_blocks'] * 5, 30)
                    predictions.append({
                        'symbol': result['symbol'],
                        'direction': result['expected_direction'],
                        'confidence': min(confidence, 90),
                        'order_blocks': result['large_blocks'],
                        'prep_volume': sum(cluster['value'] for cluster in 
                                         self._find_order_clusters([], result['expected_direction'])),
                        'timestamp': datetime.now()
                    })
            
            return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error predicting whale movements: {e}")
            return []

# Глобальний екземпляр форкастера
whale_forecaster = WhaleForecaster()