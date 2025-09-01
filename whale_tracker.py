import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class WhaleTracker:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.whale_threshold = 500000  # $500k для визначення кита
        self.suspicious_activities = []
        
    def get_large_trades(self, symbol: str = "BTCUSDT", limit: int = 100) -> List[Dict]:
        """Отримати великі угоди з останніх даних"""
        try:
            url = f"{self.base_url}/trades"
            params = {'symbol': symbol, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            trades = response.json()
            
            large_trades = []
            for trade in trades:
                trade_value = float(trade['price']) * float(trade['qty'])
                if trade_value >= self.whale_threshold:
                    large_trades.append({
                        'symbol': symbol,
                        'price': float(trade['price']),
                        'quantity': float(trade['qty']),
                        'value': trade_value,
                        'time': datetime.fromtimestamp(trade['time']/1000),
                        'is_buyer': trade['isBuyerMaker']
                    })
            
            return large_trades
        except Exception as e:
            logger.error(f"Error getting large trades: {e}")
            return []
    
    def detect_whale_accumulation(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Виявити накопичення китами"""
        try:
            # Отримуємо останні 500 угод
            large_trades = self.get_large_trades(symbol, 500)
            
            if not large_trades:
                return None
            
            # Аналізуємо покупки/продажі китів
            buy_volume = sum(trade['value'] for trade in large_trades if trade['is_buyer'])
            sell_volume = sum(trade['value'] for trade in large_trades if not trade['is_buyer'])
            
            # Перевіряємо накопичення
            if buy_volume > sell_volume * 3:  # Купівля перевищує продаж у 3 рази
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
            logger.error(f"Error detecting whale accumulation: {e}")
            return None
    
    def detect_pump_preparation(self, symbol: str) -> Optional[Dict]:
        """Виявити підготовку до пампу"""
        try:
            # Отримуємо дані про глибину ринку
            url = f"{self.base_url}/depth?symbol={symbol}&limit=50"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Аналізуємо ордери на продаж
            ask_orders = data['asks'][:20]  # Топ-20 ордерів на продаж
            large_ask_orders = []
            
            for price, quantity in ask_orders:
                order_value = float(price) * float(quantity)
                if order_value > self.whale_threshold:
                    large_ask_orders.append({
                        'price': float(price),
                        'quantity': float(quantity),
                        'value': order_value
                    })
            
            # Якщо знайдено великі ордери на продаж - можлива підготовка до пампу
            if large_ask_orders:
                total_value = sum(order['value'] for order in large_ask_orders)
                return {
                    'symbol': symbol,
                    'type': 'PUMP_PREPARATION',
                    'large_orders_count': len(large_ask_orders),
                    'total_value': total_value,
                    'orders': large_ask_orders[:5]  # Топ-5 найбільших ордерів
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pump preparation: {e}")
            return None
    
    def detect_dump_warning(self, symbol: str) -> Optional[Dict]:
        """Виявити ознаки майбутнього дампу"""
        try:
            # Отримуємо останні великі угоди
            large_trades = self.get_large_trades(symbol, 200)
            
            if not large_trades:
                return None
            
            # Шукаємо масові продажі китів
            recent_sells = [t for t in large_trades if not t['is_buyer']]
            recent_buys = [t for t in large_trades if t['is_buyer']]
            
            sell_volume = sum(t['value'] for t in recent_sells)
            buy_volume = sum(t['value'] for t in recent_buys)
            
            # Якщо обсяг продажів значно перевищує купівлю
            if sell_volume > buy_volume * 2 and len(recent_sells) > 5:
                return {
                    'symbol': symbol,
                    'type': 'DUMP_WARNING',
                    'sell_volume': sell_volume,
                    'buy_volume': buy_volume,
                    'sell_count': len(recent_sells),
                    'buy_count': len(recent_buys),
                    'ratio': sell_volume / buy_volume if buy_volume > 0 else float('inf')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting dump warning: {e}")
            return None
    
    def monitor_top_cryptos(self) -> List[Dict]:
        """Моніторинг топ-20 криптовалют"""
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'MATICUSDT', 'SHIBUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT',
            'XMRUSDT', 'ETCUSDT', 'FILUSDT', 'APTUSDT', 'NEARUSDT'
        ]
        
        alerts = []
        
        for symbol in top_symbols:
            try:
                # Перевіряємо різні типи активності
                accumulation = self.detect_whale_accumulation(symbol)
                pump_prep = self.detect_pump_preparation(symbol)
                dump_warning = self.detect_dump_warning(symbol)
                
                if accumulation:
                    alerts.append(accumulation)
                if pump_prep:
                    alerts.append(pump_prep)
                if dump_warning:
                    alerts.append(dump_warning)
                    
                # Невелика затримка між запитами
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                continue
        
        return alerts
    
    def format_whale_alert(self, alert: Dict) -> str:
        """Форматувати повідомлення про активність китів"""
        if alert['type'] == 'ACCUMULATION':
            return (f"🐋 <b>НАКОПИЧЕННЯ КИТА</b> - {alert['symbol']}\n"
                   f"📈 Обсяг купівлі: ${alert['buy_volume']:,.0f}\n"
                   f"📉 Обсяг продажу: ${alert['sell_volume']:,.0f}\n"
                   f"⚖️ Співвідношення: {alert['ratio']:.2f}x\n"
                   f"🚀 <b>Можливий майбутній PUMP</b>")
        
        elif alert['type'] == 'PUMP_PREPARATION':
            return (f"🔧 <b>ПІДГОТОВКА ДО PUMP</b> - {alert['symbol']}\n"
                   f"📊 Великих ордерів: {alert['large_orders_count']}\n"
                   f"💰 Загальна вартість: ${alert['total_value']:,.0f}\n"
                   f"⚠️ <b>Очікуйте руху ціни</b>")
        
        elif alert['type'] == 'DUMP_WARNING':
            return (f"⚠️ <b>ПОПЕРЕДЖЕННЯ ПРО DUMP</b> - {alert['symbol']}\n"
                   f"📉 Продажі китів: ${alert['sell_volume']:,.0f}\n"
                   f"📈 Купівлі: ${alert['buy_volume']:,.0f}\n"
                   f"🔻 Співвідношення: {alert['ratio']:.2f}x\n"
                   f"🎯 <b>Можливий майбутній DUMP</b>")
        
        return ""

# Глобальний екземпляр трекера
whale_tracker = WhaleTracker()