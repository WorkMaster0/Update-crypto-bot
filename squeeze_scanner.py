# squeeze_scanner.py
import requests
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from utils import safe_request

logger = logging.getLogger(__name__)

class SqueezeScanner:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def find_squeeze_opportunities(self, min_volume: float = 1000000, max_volume: float = 50000000) -> List[Dict]:
        """Знайти можливості для сквізів на малоліквідних токенах"""
        opportunities = []
        
        try:
            # Отримуємо всі USDT пари
            url = f"{self.base_url}/ticker/24hr"
            data = safe_request(url)
            
            if not data or not isinstance(data, list):
                return opportunities
            
            # Фільтруємо токени за обсягом
            low_float_symbols = [
                d for d in data 
                if isinstance(d, dict) and 
                d.get('symbol', '').endswith('USDT') and 
                min_volume < float(d.get('quoteVolume', 0)) < max_volume
            ]
            
            logger.info(f"Found {len(low_float_symbols)} low float symbols")
            
            for pair in low_float_symbols:
                symbol = pair['symbol']
                
                try:
                    # Пропускаємо стейблкоїни
                    if any(stable in symbol for stable in ['USDC', 'FDUSD', 'BUSD', 'TUSD', 'DAI']):
                        continue
                    
                    # Аналізуємо глибину ринку
                    depth_data = self.analyze_orderbook(symbol)
                    if depth_data:
                        opportunities.append(depth_data)
                    
                    # Затримка між запитами
                    time.sleep(0.05)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Сортуємо за потенційним прибутком
            opportunities.sort(key=lambda x: x.get('potential_gain_pct', 0), reverse=True)
            return opportunities[:15]  # Топ-15 opportunities
            
        except Exception as e:
            logger.error(f"Error finding squeeze opportunities: {e}")
            return []
    
    def analyze_orderbook(self, symbol: str) -> Optional[Dict]:
        """Детально аналізувати стакан токена"""
        try:
            # Отримуємо глибину ринку
            depth_url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            depth_data = safe_request(depth_url)
            
            if not depth_data or not isinstance(depth_data, dict):
                return None
            
            asks = depth_data.get('asks', [])
            bids = depth_data.get('bids', [])
            
            if len(asks) < 10 or len(bids) < 10:
                return None
            
            # Поточна ціна (середня між best bid і best ask)
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            current_price = (best_bid + best_ask) / 2
            
            if current_price <= 0:
                return None
            
            # Аналіз продажів (asks)
            ask_analysis = self.analyze_side(asks, current_price, 'ASK')
            bid_analysis = self.analyze_side(bids, current_price, 'BID')
            
            # Вибираємо кращу opportunity
            if ask_analysis['potential_gain_pct'] > bid_analysis['potential_gain_pct']:
                best_opportunity = ask_analysis
                opportunity_type = 'LONG_SQUEEZE'
            else:
                best_opportunity = bid_analysis
                opportunity_type = 'SHORT_SQUEEZE'
            
            # Додаткові метрики
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
            order_book_imbalance = sum(float(b[1]) for b in bids[:5]) / sum(float(a[1]) for a in asks[:5]) if asks and bids else 1
            
            if best_opportunity['potential_gain_pct'] > 1.5:  # Мінімум 1.5% gain
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'opportunity_type': opportunity_type,
                    'potential_gain_pct': best_opportunity['potential_gain_pct'],
                    'squeeze_cost': best_opportunity['squeeze_cost'],
                    'target_price': best_opportunity['target_price'],
                    'levels_to_break': best_opportunity['levels_to_break'],
                    'total_volume': best_opportunity['total_volume'],
                    'spread_pct': spread_pct,
                    'order_book_imbalance': order_book_imbalance,
                    'liquidity_score': self.calculate_liquidity_score(asks, bids)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing orderbook for {symbol}: {e}")
            return None
    
    def analyze_side(self, orders: List, current_price: float, side: str) -> Dict:
        """Аналізувати одну сторону стакану"""
        total_volume = 0
        total_cost = 0
        levels_to_break = 0
        target_price = current_price
        
        for i, (price, volume) in enumerate(orders[:10]):  # Перші 10 рівнів
            price_float = float(price)
            volume_float = float(volume)
            level_cost = price_float * volume_float
            
            total_volume += volume_float
            total_cost += level_cost
            levels_to_break = i + 1
            
            # Оновлюємо цільову ціну
            target_price = price_float
            
            # Якщо вартість перевищує 100K, зупиняємось
            if total_cost > 100000:
                break
        
        # Розраховуємо потенційний прибуток
        if side == 'ASK':
            potential_gain_pct = ((target_price - current_price) / current_price) * 100
        else:
            potential_gain_pct = ((current_price - target_price) / current_price) * 100
        
        return {
            'potential_gain_pct': potential_gain_pct,
            'squeeze_cost': total_cost,
            'target_price': target_price,
            'levels_to_break': levels_to_break,
            'total_volume': total_volume
        }
    
    def calculate_liquidity_score(self, asks: List, bids: List) -> float:
        """Розрахувати score ліквідності (0-100)"""
        if not asks or not bids:
            return 0
        
        # Аналізуємо глибину стакану
        top_5_ask_volume = sum(float(ask[1]) for ask in asks[:5])
        top_5_bid_volume = sum(float(bid[1]) for bid in bids[:5])
        
        # Score based on order book depth
        depth_score = min(100, (top_5_ask_volume + top_5_bid_volume) * 1000)
        
        # Score based on spread
        best_ask = float(asks[0][0])
        best_bid = float(bids[0][0])
        spread_pct = ((best_ask - best_bid) / best_bid) * 100
        spread_score = max(0, 100 - (spread_pct * 10))
        
        return (depth_score + spread_score) / 2
    
    def format_squeeze_message(self, opportunity: Dict) -> str:
        """Форматувати повідомлення про squeeze opportunity"""
        emoji = "🟢" if opportunity['opportunity_type'] == 'LONG_SQUEEZE' else "🔴"
        
        message = f"{emoji} <b>{opportunity['symbol']}</b>\n"
        message += f"   Тип: {opportunity['opportunity_type']}\n"
        message += f"   Поточна ціна: ${opportunity['current_price']:.6f}\n"
        message += f"   Цільова ціна: ${opportunity['target_price']:.6f}\n"
        message += f"   Потенційний зиск: {opportunity['potential_gain_pct']:.2f}%\n"
        message += f"   Вартість сквізу: ${opportunity['squeeze_cost']:,.0f}\n"
        message += f"   Рівнів до пробою: {opportunity['levels_to_break']}\n"
        message += f"   Спред: {opportunity['spread_pct']:.2f}%\n"
        message += f"   Score ліквідності: {opportunity['liquidity_score']:.1f}/100\n"
        
        return message

# Глобальний екземпляр сканера
squeeze_scanner = SqueezeScanner()