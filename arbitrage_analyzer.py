import requests
import logging
from typing import Dict, List, Tuple
import time

logger = logging.getLogger(__name__)

class ArbitrageAnalyzer:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.triangular_pairs = {}
        self.surface_rates = {}
        
    def get_ticker_prices(self) -> Dict[str, float]:
        """Отримати всі ціни з Binance"""
        try:
            url = f"{self.base_url}/ticker/price"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            prices = {}
            for item in data:
                prices[item['symbol']] = float(item['price'])
                
            return prices
        except Exception as e:
            logger.error(f"Error getting ticker prices: {e}")
            return {}
    
    def find_triangular_arbitrage_pairs(self, prices: Dict[str, float]) -> List[Dict]:
        """Знайти трикутні арбітражні пари"""
        # Фільтруємо тільки USDT пари
        usdt_pairs = {k: v for k, v in prices.items() if k.endswith('USDT')}
        
        # Знаходимо всі валюти, які торгуються проти USDT
        currencies = set()
        for pair in usdt_pairs.keys():
            currency = pair.replace('USDT', '')
            currencies.add(currency)
        
        # Створюємо словник цін для кожної валюти
        currency_prices = {}
        for currency in currencies:
            # Шукаємо пари між валютами (наприклад, BTCETH)
            for target_currency in currencies:
                if currency != target_currency:
                    cross_pair = f"{currency}{target_currency}"
                    if cross_pair in prices:
                        if currency not in currency_prices:
                            currency_prices[currency] = {}
                        currency_prices[currency][target_currency] = prices[cross_pair]
        
        # Знаходимо трикутні арбітражні можливості
        arbitrage_opportunities = []
        
        for currency_a in currencies:
            for currency_b in currencies:
                if currency_a != currency_b:
                    # Шукаємо шлях: A -> B -> USDT -> A
                    if (currency_a in currency_prices and 
                        currency_b in currency_prices[currency_a] and
                        f"{currency_b}USDT" in usdt_pairs and
                        f"{currency_a}USDT" in usdt_pairs):
                        
                        # Отримуємо ціни з перевіркою на наявність
                        rate_ab = currency_prices[currency_a].get(currency_b, 0)
                        if rate_ab == 0:
                            continue

                        rate_b_usdt = usdt_pairs.get(f"{currency_b}USDT", 0)
                        if rate_b_usdt == 0:
                            continue

                        usdt_a_price = usdt_pairs.get(f"{currency_a}USDT", 0)
                        if usdt_a_price == 0:
                            continue

                        rate_usdt_a = 1 / usdt_a_price
                        
                        # Загальна ставка через арбітраж
                        final_rate = rate_ab * rate_b_usdt * rate_usdt_a
                        
                        # Розраховуємо прибутковість у відсотках
                        profitability = (final_rate - 1) * 100
                        
                        if abs(profitability) > 0.1:  # Фільтруємо тільки значні можливості
                            opportunity = {
                                'path': f"{currency_a} -> {currency_b} -> USDT -> {currency_a}",
                                'profitability': profitability,
                                'rates': {
                                    f"{currency_a}/{currency_b}": rate_ab,
                                    f"{currency_b}/USDT": rate_b_usdt,
                                    f"USDT/{currency_a}": rate_usdt_a
                                },
                                'final_rate': final_rate
                            }
                            arbitrage_opportunities.append(opportunity)
        
        # Сортуємо за прибутковістю
        arbitrage_opportunities.sort(key=lambda x: abs(x['profitability']), reverse=True)
        return arbitrage_opportunities
    
    def calculate_depth_arbitrage(self, symbol: str) -> Dict:
        """Розрахувати арбітраж на основі глибини ринку"""
        try:
            # Отримуємо глибину ринку
            url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Аналізуємо найкращі ціни купівлі та продажу
            best_bid = float(data['bids'][0][0]) if data['bids'] else 0
            best_ask = float(data['asks'][0][0]) if data['asks'] else 0
            
            # Розраховуємо спред
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100 if best_bid > 0 else 0
            
            # Аналізуємо обсяги
            bid_volume = sum(float(bid[1]) for bid in data['bids'][:5])
            ask_volume = sum(float(ask[1]) for ask in data['asks'][:5])
            
            return {
                'symbol': symbol,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance': bid_volume / ask_volume if ask_volume > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating depth arbitrage for {symbol}: {e}")
            return {}
    
    def find_cross_exchange_opportunities(self, prices: Dict[str, float]) -> List[Dict]:
        """Знайти можливості для міжбіржового арбітражу (теоретично)"""
        # Ця функція може бути розширена для роботи з іншими біржами
        # Поки що просто повертаємо порожній список
        return []
    
    def format_opportunity_message(self, opportunity: Dict) -> str:
        """Форматувати повідомлення про арбітражну можливість"""
        profit = opportunity['profitability']
        profit_emoji = "🟢" if profit > 0 else "🔴"
        
        message = f"{profit_emoji} <b>Арбітражна можливість</b>\n"
        message += f"Шлях: {opportunity['path']}\n"
        message += f"Прибутковість: <b>{profit:+.4f}%</b>\n"
        message += f"Фінальний курс: {opportunity['final_rate']:.8f}\n"
        
        # Додаємо деталі по курсам
        for pair, rate in opportunity['rates'].items():
            message += f"{pair}: {rate:.8f}\n"
            
        return message

# Глобальний екземпляр аналізатора
arbitrage_analyzer = ArbitrageAnalyzer()