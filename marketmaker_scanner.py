# marketmaker_scanner.py
import requests
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from utils import safe_request

logger = logging.getLogger(__name__)

class MarketMakerScanner:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_current_price(self, symbol: str) -> float:
        """Отримати поточну ціну токена"""
        try:
            url = f"{self.base_url}/ticker/price?symbol={symbol}"
            data = safe_request(url)
            if data and isinstance(data, dict) and 'price' in data:
                return float(data['price'])
            return 0
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0
        
    def find_marketmaker_anomalies(self, symbol: str, bids: List, asks: List) -> List[Dict]:
        """Знайти аномалії маркетмейкерів в стакані"""
        anomalies = []
        
        try:
            if not bids or not asks or len(bids) < 10 or len(asks) < 10:
                return anomalies
            
            # Отримуємо поточну ціну для перевірки
            current_price = self.get_current_price(symbol)
            
            # Конвертуємо ціни в числа
            bid_prices = [float(b[0]) for b in bids]
            ask_prices = [float(a[0]) for a in asks]
            
            # 1. Пошук пропусків ліквідності
            for i in range(1, min(20, len(bid_prices))):
                if bid_prices[i-1] <= bid_prices[i]:
                    continue
                    
                gap = (bid_prices[i-1] - bid_prices[i]) / bid_prices[i] * 100
                if gap > 0.3:  # Пропуск більше 0.3%
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'LIQUIDITY_GAP',
                        'price': bid_prices[i],
                        'volume': float(bids[i][1]),
                        'impact': round(gap, 2),
                        'side': 'BID',
                        'opportunity': 'BUY_LIMIT_IN_GAP'
                    })
            
            for i in range(1, min(20, len(ask_prices))):
                if ask_prices[i] <= ask_prices[i-1]:
                    continue
                    
                gap = (ask_prices[i] - ask_prices[i-1]) / ask_prices[i-1] * 100
                if gap > 0.3:
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'LIQUIDITY_GAP',
                        'price': ask_prices[i],
                        'volume': float(asks[i][1]),
                        'impact': round(gap, 2),
                        'side': 'ASK',
                        'opportunity': 'SELL_LIMIT_IN_GAP'
                    })
            
            # 2. Пошук fat-finger ордерів (дуже великі) з перевіркою ціни
            for i, (price, volume) in enumerate(bids[:15]):
                order_size = float(price) * float(volume)
                price_float = float(price)
                
                # Перевіряємо чи ціна в межах ±10% від поточної
                if (order_size > 200000 and current_price > 0 and
                    0.9 * current_price < price_float < 1.1 * current_price):
                    
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'FAT_FINGER',
                        'price': price_float,
                        'volume': float(volume),
                        'size_usd': order_size,
                        'side': 'BID',
                        'position': i + 1,
                        'opportunity': 'WAIT_FOR_CANCEL'
                    })
            
            for i, (price, volume) in enumerate(asks[:15]):
                order_size = float(price) * float(volume)
                price_float = float(price)
                
                if (order_size > 200000 and current_price > 0 and
                    0.9 * current_price < price_float < 1.1 * current_price):
                    
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'FAT_FINGER',
                        'price': price_float,
                        'volume': float(volume),
                        'size_usd': order_size,
                        'side': 'ASK', 
                        'position': i + 1,
                        'opportunity': 'WAIT_FOR_CANCEL'
                    })
            
            # 3. Пошук маніпулятивних стін з перевіркою ціни
            for i, (price, volume) in enumerate(bids[:5]):
                order_size = float(price) * float(volume)
                price_float = float(price)
                
                if (order_size > 500000 and i == 0 and current_price > 0 and
                    0.9 * current_price < price_float < 1.1 * current_price):
                    
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'MANIPULATION_WALL',
                        'price': price_float,
                        'volume': float(volume),
                        'size_usd': order_size,
                        'side': 'BID',
                        'opportunity': 'FOLLOW_WHALE'
                    })
            
            for i, (price, volume) in enumerate(asks[:5]):
                order_size = float(price) * float(volume)
                price_float = float(price)
                
                if (order_size > 500000 and i == 0 and current_price > 0 and
                    0.9 * current_price < price_float < 1.1 * current_price):
                    
                    anomalies.append({
                        'symbol': symbol,
                        'type': 'MANIPULATION_WALL',
                        'price': price_float,
                        'volume': float(volume),
                        'size_usd': order_size,
                        'side': 'ASK',
                        'opportunity': 'FOLLOW_WHALE'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error analyzing market maker anomalies for {symbol}: {e}")
            return []
    
    def scan_top_symbols(self, top_count: int = 20) -> List[Dict]:
        """Сканування топових символів на аномалії"""
        all_anomalies = []
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'MATICUSDT', 'SHIBUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT',
            'XMRUSDT', 'ETCUSDT', 'FILUSDT', 'APTUSDT', 'NEARUSDT'
        ]
        
        for symbol in top_symbols[:top_count]:
            try:
                # Отримуємо глибину ринку
                depth_url = f"{self.base_url}/depth?symbol={symbol}&limit=50"
                depth_data = safe_request(depth_url)
                
                if not depth_data or not isinstance(depth_data, dict):
                    continue
                
                bids = depth_data.get('bids', [])
                asks = depth_data.get('asks', [])
                
                # Шукаємо аномалії
                anomalies = self.find_marketmaker_anomalies(symbol, bids, asks)
                if anomalies:
                    all_anomalies.extend(anomalies)
                
                # Невелика затримка між запитами
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Сортуємо за потенційним impact
        all_anomalies.sort(key=lambda x: x.get('impact', 0) if 'impact' in x else x.get('size_usd', 0), reverse=True)
        return all_anomalies
    
    def format_anomaly_message(self, anomaly: Dict) -> str:
        """Форматувати повідомлення про аномалію"""
        if anomaly['type'] == 'LIQUIDITY_GAP':
            return (f"📊 <b>Пропуск ліквідності</b> - {anomaly['symbol']}\n"
                   f"Сторона: {anomaly['side']}\n"
                   f"Ціна: ${anomaly['price']:.6f}\n"
                   f"Розрив: {anomaly['impact']}%\n"
                   f"🎯 Стратегія: {anomaly['opportunity']}")
        
        elif anomaly['type'] == 'FAT_FINGER':
            return (f"💥 <b>Fat-Finger ордер</b> - {anomaly['symbol']}\n"
                   f"Сторона: {anomaly['side']}\n"
                   f"Ціна: ${anomaly['price']:.6f}\n"
                   f"Розмір: ${anomaly['size_usd']:,.0f}\n"
                   f"Позиція: #{anomaly['position']}\n"
                   f"🎯 Стратегія: {anomaly['opportunity']}")
        
        elif anomaly['type'] == 'MANIPULATION_WALL':
            return (f"🎭 <b>Маніпулятивна стіна</b> - {anomaly['symbol']}\n"
                   f"Сторона: {anomaly['side']}\n"
                   f"Ціна: ${anomaly['price']:.6f}\n"
                   f"Розмір: ${anomaly['size_usd']:,.0f}\n"
                   f"🎯 Стратегія: {anomaly['opportunity']}")
        
        return ""

# Глобальний екземпляр сканера
marketmaker_scanner = MarketMakerScanner()