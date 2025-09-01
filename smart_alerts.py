python
import logging
from datetime import datetime
from typing import Dict, List, Callable
import json
logger = logging.getLogger(__name__)
class SmartAlertSystem:
    def __init__(self):
        self.user_alerts = {}  # user_id -> list of alerts
        self.conditions = {
            'price_above': self.check_price_above,
            'price_below': self.check_price_below,
            'volume_spike': self.check_volume_spike,
            'rsi_overbought': self.check_rsi_overbought,
            'rsi_oversold': self.check_rsi_oversold,
            'price_change_24h': self.check_price_change_24h
        }
    
    def create_alert(self, user_id: int, symbol: str, condition: str, value: float, alert_message: str) -> Dict:
        """Створити нове сповіщення"""
        alert_id = datetime.now().timestamp()
        
        alert = {
            'id': alert_id,
            'symbol': symbol,
            'condition': condition,
            'value': value,
            'message': alert_message,
            'active': True
        }
        
        if user_id not in self.user_alerts:
            self.user_alerts[user_id] = []
        
        self.user_alerts[user_id].append(alert)
        return alert
    
    def check_price_above(self, current_price: float, alert_value: float) -> bool:
        return current_price > alert_value
    
    def check_price_below(self, current_price: float, alert_value: float) -> bool:
        return current_price < alert_value
    
    def check_volume_spike(self, current_volume: float, avg_volume: float, alert_value: float) -> bool:
        return current_volume > avg_volume * alert_value
    
    def check_rsi_overbought(self, rsi: float, alert_value: float) -> bool:
        return rsi > alert_value
    
    def check_rsi_oversold(self, rsi: float, alert_value: float) -> bool:
        return rsi < alert_value
    
    def check_price_change_24h(self, price_change: float, alert_value: float) -> bool:
        return abs(price_change) > alert_value
    
    def check_alerts(self, user_id: int, market_data: Dict) -> List[Dict]:
        """Перевірити всі активні сповіщення користувача"""
        triggered_alerts = []
        
        if user_id not in self.user_alerts:
            return triggered_alerts
        
        for alert in self.user_alerts[user_id]:
            if not alert['active']:
                continue
            
            symbol = alert['symbol']
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            condition = alert['condition']
            
            try:
                if condition == 'price_above':
                    triggered = self.check_price_above(data['price'], alert['value'])
                elif condition == 'price_below':
                    triggered = self.check_price_below(data['price'], alert['value'])
                elif condition == 'volume_spike':
                    triggered = self.check_volume_spike(data['volume'], data['avg_volume'], alert['value'])
                elif condition == 'rsi_overbought':
                    triggered = self.check_rsi_overbought(data['rsi'], alert['value'])
                elif condition == 'rsi_oversold':
                    triggered = self.check_rsi_oversold(data['rsi'], alert['value'])
                elif condition == 'price_change_24h':
                    triggered = self.check_price_change_24h(data['price_change_24h'], alert['value'])
                else:
                    continue
                
                if triggered:
                    triggered_alerts.append(alert)
                    alert['active'] = False  # Деактивувати спрацьований alert
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert['id']}: {e}")
                continue
        
        return triggered_alerts
# Глобальний екземпляр системи сповіщень
alert_system = SmartAlertSystem()