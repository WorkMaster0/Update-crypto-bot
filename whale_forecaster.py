# whale_forecaster.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
from whale_analyzer import whale_analyzer
from utils import safe_request

logger = logging.getLogger(__name__)

class WhaleForecaster:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.orderbook_snapshots = {}
        
    def analyze_orderbook_patterns(self, symbol: str) -> Optional[Dict]:
        """Аналіз патернів у стакані для передбачення китових рухів"""
        try:
            depth = safe_request(f"{self.base_url}/depth?symbol={symbol}&limit=100")
            if not depth or not isinstance(depth, dict):
                return None
            
            bids = depth.get('bids', [])
            asks = depth.get('asks', [])
            
            # Аналізуємо кластеризацію ордерів
            bid_clusters = self._find_order_clusters(bids, 'bid')
            ask_clusters = self._find_order_clusters(asks, 'ask')
            
            # Визначаємо потенційний напрямок
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
            logger.error(f"Orderbook pattern analysis failed for {symbol}: {e}")
            return None
    
    def _find_order_clusters(self, orders: List, order_type: str) -> List[Dict]:
        """Знаходження кластерів великих ордерів"""
        clusters = []
        min_cluster_size = 50000  # $50k мінімальний розмір кластера
        
        for i, order in enumerate(orders):
            if len(order) >= 2:
                try:
                    price = float(order[0])
                    quantity = float(order[1])
                    value = price * quantity
                    
                    if value >= min_cluster_size:
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
        """Визначення потенційного напрямку на основі кластерів"""
        if not bid_clusters and not ask_clusters:
            return None
        
        bid_pressure = sum(cluster['value'] for cluster in bid_clusters)
        ask_pressure = sum(cluster['value'] for cluster in ask_clusters)
        
        if bid_pressure > ask_pressure * 1.5:
            return 'BUY'
        elif ask_pressure > bid_pressure * 1.5:
            return 'SELL'
        
        return None
    
    def detect_shadow_orders(self, symbol: str) -> Optional[Dict]:
        """Виявлення 'тіневих' ордерів китів (з'явлення/зникнення великих ордерів)"""
        try:
            current_time = datetime.now()
            current_depth = safe_request(f"{self.base_url}/depth?symbol={symbol}&limit=100")
            
            if symbol not in self.orderbook_snapshots:
                self.orderbook_snapshots[symbol] = {
                    'depth': current_depth,
                    'timestamp': current_time
                }
                return None
            
            previous_depth = self.orderbook_snapshots[symbol]['depth']
            previous_time = self.orderbook_snapshots[symbol]['timestamp']
            
            # Перевіряємо чи минуло достатньо часу (5-10 секунд)
            if (current_time - previous_time).total_seconds() < 5:
                return None
            
            # Аналізуємо зміни
            changes = self._analyze_orderbook_changes(previous_depth, current_depth)
            
            # Оновлюємо snapshot
            self.orderbook_snapshots[symbol] = {
                'depth': current_depth,
                'timestamp': current_time
            }
            
            return changes if changes else None
            
        except Exception as e:
            logger.error(f"Shadow orders detection failed for {symbol}: {e}")
            return None
    
    def _analyze_orderbook_changes(self, old_depth: Dict, new_depth: Dict) -> Optional[Dict]:
        """Аналіз змін у стакані за останні 5-10 секунд"""
        try:
            old_bids = old_depth.get('bids', [])
            new_bids = new_depth.get('bids', [])
            old_asks = old_depth.get('asks', [])
            new_asks = new_depth.get('asks', [])
            
            # Знаходимо нові великі ордери, що з'явилися
            new_large_bids = self._find_new_large_orders(old_bids, new_bids, 'bid')
            new_large_asks = self._find_new_large_orders(old_asks, new_asks, 'ask')
            
            # Знаходимо великі ордери, що зникли
            disappeared_bids = self._find_disappeared_orders(old_bids, new_bids, 'bid')
            disappeared_asks = self._find_disappeared_orders(old_asks, new_asks, 'ask')
            
            total_value = (sum(order['value'] for order in new_large_bids + new_large_asks + 
                             disappeared_bids + disappeared_asks))
            
            if total_value > 100000:  # Мінімум $100k змін
                return {
                    'new_bids': new_large_bids,
                    'new_asks': new_large_asks,
                    'disappeared_bids': disappeared_bids,
                    'disappeared_asks': disappeared_asks,
                    'total_value': total_value,
                    'change_count': len(new_large_bids + new_large_asks + 
                                      disappeared_bids + disappeared_asks)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Orderbook changes analysis failed: {e}")
            return None
    
    def _find_new_large_orders(self, old_orders: List, new_orders: List, order_type: str) -> List[Dict]:
        """Знаходження нових великих ордерів"""
        new_orders_dict = {order[0]: float(order[1]) for order in new_orders if len(order) >= 2}
        old_orders_dict = {order[0]: float(order[1]) for order in old_orders if len(order) >= 2}
        
        large_orders = []
        for price, quantity in new_orders_dict.items():
            if price not in old_orders_dict and quantity * float(price) > 50000:  # $50k+
                large_orders.append({
                    'price': float(price),
                    'quantity': quantity,
                    'value': float(price) * quantity,
                    'type': order_type
                })
        
        return large_orders
    
    def _find_disappeared_orders(self, old_orders: List, new_orders: List, order_type: str) -> List[Dict]:
        """Знаходження ордерів, що зникли"""
        return self._find_new_large_orders(new_orders, old_orders, order_type)
    
    def calculate_prediction_confidence(self, orderbook_analysis: Dict, shadow_analysis: Dict) -> float:
        """Розрахунок впевненості у передбаченні"""
        confidence = 50.0  # Базова впевненість
        
        # Додаємо бали за кількість кластерів
        confidence += min(orderbook_analysis['large_blocks'] * 5, 20)
        
        # Додаємо бали за обсяг змін
        confidence += min(shadow_analysis['total_value'] / 10000, 25)
        
        # Додаємо бали за кількість змін
        confidence += min(shadow_analysis['change_count'] * 3, 15)
        
        return min(confidence, 95.0)  # Максимум 95%
    
    def predict_whale_movements(self) -> List[Dict]:
        """Передбачення майбутніх китових активностей"""
        try:
            high_volume_symbols = whale_analyzer.get_high_volume_symbols()
            predictions = []
            
            for symbol in high_volume_symbols[:8]:  # Аналізуємо топ-8
                # Аналіз ордербуків на глибині
                orderbook = self.analyze_orderbook_patterns(symbol)
                
                # Виявлення "тіней китів"
                shadow_orders = self.detect_shadow_orders(symbol)
                
                if orderbook and shadow_orders:
                    confidence = self.calculate_prediction_confidence(orderbook, shadow_orders)
                    
                    if confidence > 65:  # Якщо впевненість >65%
                        predictions.append({
                            'symbol': symbol,
                            'direction': orderbook['expected_direction'],
                            'confidence': round(confidence, 1),
                            'order_blocks': orderbook['large_blocks'],
                            'prep_volume': shadow_orders['total_value'],
                            'timestamp': datetime.now()
                        })
            
            return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error predicting whale movements: {e}")
            return []

# Глобальний екземпляр форкастера
whale_forecaster = WhaleForecaster()