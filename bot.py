import os
import requests
import logging
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler

# Налаштування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("BOT_TOKEN не знайдено в змінних оточення")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

# ==================== TRADE ASSISTANT CLASS ====================
class TradeAssistant:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_market_data(self, symbol: str):
        try:
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
    
    def generate_trade_signal(self, symbol: str):
        market_data = self.get_market_data(symbol)
        if not market_data:
            return {'error': 'Could not fetch market data'}
        
        trend_analysis = self.analyze_trend(market_data['klines'])
        volume_analysis = self.analyze_volume(market_data['klines'])
        momentum_analysis = self.analyze_momentum(market_data['klines'])
        liquidity_analysis = self.analyze_liquidity(market_data['depth'])
        
        recommendation = self.generate_recommendation(
            trend_analysis, volume_analysis, momentum_analysis, liquidity_analysis
        )
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': self.calculate_confidence(trend_analysis, volume_analysis, momentum_analysis),
            'entry_points': self.calculate_entry_points(market_data['klines']),
            'exit_points': self.calculate_exit_points(market_data['klines']),
            'risk_level': self.calculate_risk_level(market_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_trend(self, klines):
        closes = [float(k[4]) for k in klines]
        price_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] != 0 else 0
        
        return {
            'direction': 'up' if price_change > 0 else 'down',
            'strength': abs(price_change),
            'trend_type': self.determine_trend_type(closes)
        }
    
    def analyze_volume(self, klines):
        volumes = [float(k[5]) for k in klines]
        current_volume = volumes[-1] if volumes else 0
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else current_volume
        
        return {
            'current_volume': current_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'volume_trend': 'increasing' if current_volume > avg_volume else 'decreasing'
        }
    
    def analyze_momentum(self, klines):
        closes = [float(k[4]) for k in klines]
        rsi = self.calculate_rsi(closes)
        
        return {
            'rsi': rsi,
            'momentum': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
            'price_acceleration': self.calculate_acceleration(closes)
        }
    
    def analyze_liquidity(self, depth):
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
    
    def generate_recommendation(self, trend, volume, momentum, liquidity):
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
    
    def calculate_confidence(self, trend, volume, momentum):
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
    
    def calculate_entry_points(self, klines):
        closes = [float(k[4]) for k in klines]
        current_price = closes[-1] if closes else 0
        
        return [
            current_price * 0.98,
            current_price * 0.95, 
            current_price * 0.92
        ]
    
    def calculate_exit_points(self, klines):
        closes = [float(k[4]) for k in klines]
        current_price = closes[-1] if closes else 0
        
        return [
            current_price * 1.05,
            current_price * 1.08,
            current_price * 1.12
        ]
    
    def calculate_risk_level(self, market_data):
        volatility = self.calculate_volatility([float(k[4]) for k in market_data['klines']])
        
        if volatility > 10:
            return "HIGH"
        elif volatility > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Допоміжні функції
    def get_klines(self, symbol: str, interval: str, limit: int):
        try:
            url = f"{self.base_url}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except:
            return None
    
    def get_ticker_24hr(self, symbol: str):
        try:
            url = f"{self.base_url}/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=10)
            return response.json()
        except:
            return None
    
    def get_depth(self, symbol: str):
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            response = requests.get(url, timeout=10)
            return response.json()
        except:
            return None
    
    def calculate_rsi(self, prices, period: int = 14):
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
    
    def calculate_volatility(self, prices):
        if len(prices) < 2:
            return 0
            
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return sum(abs(r) for r in returns) / len(returns) * 100
    
    def determine_trend_type(self, prices):
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
    
    def calculate_acceleration(self, prices):
        if len(prices) < 3:
            return 0
            
        recent_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        previous_change = (prices[-2] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
        
        return (recent_change - previous_change) * 100
    
    def calculate_spread_percentage(self, bids, asks):
        if not bids or not asks:
            return 0
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        return ((best_ask - best_bid) / best_bid) * 100 if best_bid != 0 else 0
    
    def calculate_imbalance(self, bids, asks):
        if not bids or not asks:
            return 1
            
        bid_volume = sum(float(bid[1]) for bid in bids[:3])
        ask_volume = sum(float(ask[1]) for ask in asks[:3])
        
        return bid_volume / ask_volume if ask_volume > 0 else float('inf')

# ==================== ARBITRAGE ANALYZER CLASS ====================
class ArbitrageAnalyzer:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_ticker_prices(self):
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
    
    def find_triangular_arbitrage_pairs(self, prices):
        usdt_pairs = {k: v for k, v in prices.items() if k.endswith('USDT')}
        
        currencies = set()
        for pair in usdt_pairs.keys():
            currency = pair.replace('USDT', '')
            currencies.add(currency)
        
        currency_prices = {}
        for currency in currencies:
            for target_currency in currencies:
                if currency != target_currency:
                    cross_pair = f"{currency}{target_currency}"
                    if cross_pair in prices:
                        if currency not in currency_prices:
                            currency_prices[currency] = {}
                        currency_prices[currency][target_currency] = prices[cross_pair]
        
        arbitrage_opportunities = []
        
        for currency_a in currencies:
            for currency_b in currencies:
                if currency_a != currency_b:
                    if (currency_a in currency_prices and 
                        currency_b in currency_prices[currency_a] and
                        f"{currency_b}USDT" in usdt_pairs and
                        f"{currency_a}USDT" in usdt_pairs):
                        
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
                        
                        final_rate = rate_ab * rate_b_usdt * rate_usdt_a
                        profitability = (final_rate - 1) * 100
                        
                        if abs(profitability) > 0.1:
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
        
        arbitrage_opportunities.sort(key=lambda x: abs(x['profitability']), reverse=True)
        return arbitrage_opportunities
    
    def calculate_depth_arbitrage(self, symbol: str):
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            best_bid = float(data['bids'][0][0]) if data['bids'] else 0
            best_ask = float(data['asks'][0][0]) if data['asks'] else 0
            
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100 if best_bid > 0 else 0
            
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
    
    def format_opportunity_message(self, opportunity: dict) -> str:
        profit = opportunity['profitability']
        profit_emoji = "🟢" if profit > 0 else "🔴"
        
        message = f"{profit_emoji} <b>Арбітражна можливість</b>\n"
        message += f"Шлях: {opportunity['path']}\n"
        message += f"Прибутковість: <b>{profit:+.4f}%</b>\n"
        message += f"Фінальний курс: {opportunity['final_rate']:.8f}\n"
        
        for pair, rate in opportunity['rates'].items():
            message += f"{pair}: {rate:.8f}\n"
            
        return message

# ==================== WHALE TRACKER CLASS ====================
class WhaleTracker:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.whale_threshold = 500000
        
    def get_large_trades(self, symbol: str = "BTCUSDT", limit: int = 100):
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
    
    def detect_whale_accumulation(self, symbol: str = "BTCUSDT"):
        try:
            large_trades = self.get_large_trades(symbol, 500)
            
            if not large_trades:
                return None
            
            buy_volume = sum(trade['value'] for trade in large_trades if trade['is_buyer'])
            sell_volume = sum(trade['value'] for trade in large_trades if not trade['is_buyer'])
            
            if buy_volume > sell_volume * 3:
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
    
    def detect_pump_preparation(self, symbol: str):
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=50"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            ask_orders = data['asks'][:20]
            large_ask_orders = []
            
            for price, quantity in ask_orders:
                order_value = float(price) * float(quantity)
                if order_value > self.whale_threshold:
                    large_ask_orders.append({
                        'price': float(price),
                        'quantity': float(quantity),
                        'value': order_value
                    })
            
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
            logger.error(f"Error detecting pump preparation: {e}")
            return None
    
    def detect_dump_warning(self, symbol: str):
        try:
            large_trades = self.get_large_trades(symbol, 200)
            
            if not large_trades:
                return None
            
            recent_sells = [t for t in large_trades if not t['is_buyer']]
            recent_buys = [t for t in large_trades if t['is_buyer']]
            
            sell_volume = sum(t['value'] for t in recent_sells)
            buy_volume = sum(t['value'] for t in recent_buys)
            
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
    
    def monitor_top_cryptos(self):
        top_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
        ]
        
        alerts = []
        
        for symbol in top_symbols:
            try:
                accumulation = self.detect_whale_accumulation(symbol)
                pump_prep = self.detect_pump_preparation(symbol)
                dump_warning = self.detect_dump_warning(symbol)
                
                if accumulation:
                    alerts.append(accumulation)
                if pump_prep:
                    alerts.append(pump_prep)
                if dump_warning:
                    alerts.append(dump_warning)
                    
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                continue
        
        return alerts
    
    def format_whale_alert(self, alert: dict) -> str:
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

# ==================== GLOBAL VARIABLES ====================
USER_SETTINGS = {
    'min_volume': 5000000,
    'top_symbols': 30,
    'window_size': 20,
    'sensitivity': 0.005,
    'pump_threshold': 15,
    'dump_threshold': -15,
    'volume_spike_multiplier': 2.0,
    'rsi_overbought': 70,
    'rsi_oversold': 30
}

ALERT_SUBSCRIPTIONS = {}
trade_assistant = TradeAssistant()
arbitrage_analyzer = ArbitrageAnalyzer()
whale_tracker = WhaleTracker()

# ==================== HELPER FUNCTIONS ====================
def get_klines(symbol, interval="1h", limit=200):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = requests.get(url, params=params, timeout=10).json()
        
        if not data:
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

def find_support_resistance(prices, window=20, delta=0.005):
    n = len(prices)
    rolling_high = [0] * n
    rolling_low = [0] * n
    
    for i in range(window, n):
        rolling_high[i] = max(prices[i-window:i])
        rolling_low[i] = min(prices[i-window:i])
    
    levels = []
    for i in range(window, n):
        if prices[i] >= rolling_high[i] * (1 - delta):
            levels.append(rolling_high[i])
        elif prices[i] <= rolling_low[i] * (1 + delta):
            levels.append(rolling_low[i])
    
    return sorted(set(levels))

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_volume_spike(volumes, lookback=20):
    if len(volumes) < lookback:
        return False
    recent_volume = volumes[-1]
    avg_volume = sum(volumes[-lookback:]) / lookback
    return recent_volume > USER_SETTINGS['volume_spike_multiplier'] * avg_volume

def calculate_technical_indicators(closes, volumes):
    rsi = calculate_rsi(closes)
    vol_spike = calculate_volume_spike(volumes)
    return rsi, vol_spike

def detect_pump_dump(closes, volumes, pump_threshold=15, dump_threshold=-15):
    if len(closes) < 24:
        return None, 0
    
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    vol_spike = calculate_volume_spike(volumes)
    
    event_type = None
    if price_change_24h > pump_threshold and vol_spike:
        event_type = "PUMP"
    elif price_change_24h < dump_threshold and vol_spike:
        event_type = "DUMP"
    
    return event_type, price_change_24h

def detect_pump_activity(symbol, closes, volumes, settings):
    if len(closes) < 24:
        return None, 0, {}
    
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    price_change_1h = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
    
    volume_metrics = analyze_volume(volumes, settings)
    volatility = calculate_volatility(closes[-24:])
    green_candles = count_green_candles(closes[-24:])
    
    is_pump = (
        price_change_24h > settings['pump_threshold'] and
        volume_metrics['volume_spike'] and
        price_change_1h > 5 and
        green_candles > 15
    )
    
    if not is_pump:
        return None, price_change_24h, volume_metrics
    
    risk_level = calculate_pump_risk(closes, volumes, price_change_24h)
    
    pump_data = {
        'risk_level': risk_level,
        '1h_change': price_change_1h,
        'volatility': volatility,
        'green_candles': green_candles,
        'volume_metrics': volume_metrics
    }
    
    return "PUMP", price_change_24h, pump_data

def analyze_volume(volumes, settings):
    if len(volumes) < 24:
        return {'volume_spike': False, 'avg_volume': 0}
    
    current_volume = volumes[-1]
    avg_volume_24h = sum(volumes[-24:]) / 24
    avg_volume_7d = sum(volumes[-168:]) / 168 if len(volumes) >= 168 else avg_volume_24h
    
    volume_spike = current_volume > avg_volume_24h * settings['volume_spike_multiplier']
    volume_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 0
    
    return {
        'volume_spike': volume_spike,
        'avg_volume_24h': avg_volume_24h,
        'avg_volume_7d': avg_volume_7d,
        'volume_ratio': volume_ratio,
        'current_volume': current_volume
    }

def calculate_volatility(prices):
    if len(prices) < 2:
        return 0
    
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    return sum(abs(r) for r in returns) / len(returns) * 100

def count_green_candles(prices):
    if len(prices) < 2:
        return 0
    
    green_count = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            green_count += 1
    
    return green_count

def calculate_pump_risk(closes, volumes, price_change):
    risk = 5
    
    if price_change > 50:
        risk += 3
    elif price_change > 30:
        risk += 2
    elif price_change > 15:
        risk += 1
    
    if len(volumes) > 0:
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if sum(volumes[-10:]) > 0 else 1
        if volume_ratio > 5:
            risk += 2
        elif volume_ratio > 3:
            risk += 1
    
    return max(1, min(10, risk))

def detect_volume_anomaly(symbol, volumes, settings):
    if len(volumes) < 24:
        return False, {}
    
    current_volume = volumes[-1]
    avg_volume_24h = sum(volumes[-24:]) / 24
    
    is_anomaly = current_volume > avg_volume_24h * settings['volume_spike_multiplier'] * 1.5
    
    if not is_anomaly:
        return False, {}
    
    anomaly_data = {
        'current_volume': current_volume,
        'avg_volume_24h': avg_volume_24h,
        'volume_ratio': current_volume / avg_volume_24h,
        'anomaly_type': 'VOLUME_SPIKE'
    }
    
    return True, anomaly_data

def send_alerts_to_subscribers():
    if not ALERT_SUBSCRIPTIONS:
        return
    
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x.get("priceChangePercent", 0))),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:USER_SETTINGS['top_symbols']]]
        alerts = []
        
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]

                event_type, price_change = detect_pump_dump(closes, volumes)
                
                if event_type:
                    alert_text = (
                        f"🔴 {event_type} DETECTED!\n"
                        f"Токен: {symbol}\n"
                        f"Зміна ціни: {price_change:+.1f}%\n"
                        f"Рекомендація: {'Шорт' if event_type == 'PUMP' else 'Лонг'}"
                    )
                    alerts.append(alert_text)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if alerts:
            alert_text = "\n\n".join(alerts[:3])
            
            for chat_id in ALERT_SUBSCRIPTIONS.keys():
                try:
                    bot.send_message(chat_id, f"🚨 АВТОМАТИЧНЕ СПОВІЩЕННЯ:\n\n{alert_text}")
                except Exception as e:
                    logger.error(f"Error sending alert to {chat_id}: {e}")
                    
    except Exception as e:
        logger.error(f"Error in alert system: {e}")

# ==================== SCHEDULER ====================
scheduler = BackgroundScheduler()
scheduler.add_job(send_alerts_to_subscribers, 'interval', minutes=30)
scheduler.start()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return "Crypto Bot is running!"

# ==================== BOT COMMANDS ====================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = """
🤖 Smart Crypto Bot - Розширений аналіз ринку

🚀 <b>НОВІ КОМАНДИ:</b>
/trade_signal <token> - Генерація торгових сигналів
/whale_alert - Моніторинг китової активності
/arbitrage - Пошук арбітражних можливостей
/market_depth <pair> - Аналіз глибини ринку

📊 <b>Основні команди:</b>
/smart_auto - Автоматичний пошук сигналів
/pump_scan - Сканування на памп активність  
/volume_anomaly - Пошук аномальних обсягів
/advanced_analysis <token> - Розширений аналіз токена

⚙️ <b>Інші команди:</b>
/settings - Налаштування
/check_token <token> - Перевірка токена
/stats - Статистика ринку
/alerts_on - Увімкнути сповіщення
/alerts_off - Вимкнути сповіщення
"""
    bot.reply_to(message, help_text, parse_mode="HTML")

@bot.message_handler(commands=['alerts_on'])
def enable_alerts(message):
    ALERT_SUBSCRIPTIONS[message.chat.id] = True
    bot.reply_to(message, "🔔 Сповіщення увімкнено! Ви отримуватимете автоматичні сповіщення про памп/дамп.")

@bot.message_handler(commands=['alerts_off'])
def disable_alerts(message):
    if message.chat.id in ALERT_SUBSCRIPTIONS:
        del ALERT_SUBSCRIPTIONS[message.chat.id]
    bot.reply_to(message, "🔕 Сповіщення вимкнено.")

@bot.message_handler(commands=['pump_scan'])
def pump_scan_handler(message):
    try:
        msg = bot.send_message(message.chat.id, "🔍 Сканую на памп активність...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and 
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x.get("priceChangePercent", 0))),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:USER_SETTINGS['top_symbols']]]
        pump_signals = []
        
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue
                
                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]
                
                pump_type, price_change, pump_data = detect_pump_activity(
                    symbol, closes, volumes, USER_SETTINGS
                )
                
                if pump_type == "PUMP":
                    risk_level = pump_data.get('risk_level', 5)
                    risk_emoji = "🔴" if risk_level > 7 else "🟡" if risk_level > 5 else "🟢"
                    
                    signal_text = (
                        f"{risk_emoji} <b>{symbol}</b>\n"
                        f"📈 Зміна ціни: {price_change:+.1f}%\n"
                        f"⚠️ Рівень ризику: {risk_level}/10\n"
                        f"📊 Волатильність: {pump_data.get('volatility', 0):.1f}%\n"
                        f"🟢 Зелені свічки: {pump_data.get('green_candles', 0)}/24\n"
                        f"💹 Співвідношення обсягу: {pump_data.get('volume_metrics', {}).get('volume_ratio', 0):.1f}x\n"
                    )
                    
                    if risk_level > 7:
                        signal_text += "🔻 Високий ризик корекції!\n"
                    
                    pump_signals.append(signal_text)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not pump_signals:
            bot.edit_message_text("ℹ️ Пампи не знайдено.", message.chat.id, msg.message_id)
        else:
            text = "<b>🚨 Результати сканування пампа:</b>\n\n" + "\n".join(pump_signals[:5])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in pump_scan: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['volume_anomaly'])
def volume_anomaly_handler(message):
    try:
        msg = bot.send_message(message.chat.id, "🔍 Шукаю аномальні обсяги...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if isinstance(d, dict) and 
            d.get("symbol", "").endswith("USDT") and 
            float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume'] / 10
        ]

        symbols = sorted(
            symbols,
            key=lambda x: float(x.get("quoteVolume", 0)),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:50]]
        anomalies = []
        
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=100)
                if not df or len(df.get("v", [])) < 24:
                    continue
                
                volumes = [float(v) for v in df["v"]]
                
                is_anomaly, anomaly_data = detect_volume_anomaly(symbol, volumes, USER_SETTINGS)
                
                if is_anomaly:
                    anomaly_text = (
                        f"📊 <b>{symbol}</b>\n"
                        f"💥 Поточний обсяг: {anomaly_data.get('current_volume', 0):.0f}\n"
                        f"📈 Середній обсяг: {anomaly_data.get('avg_volume_24h', 0):.0f}\n"
                        f"🚀 Співвідношення: {anomaly_data.get('volume_ratio', 0):.1f}x\n"
                    )
                    anomalies.append(anomaly_text)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not anomalies:
            bot.edit_message_text("ℹ️ Аномалій обсягу не знайдено.", message.chat.id, msg.message_id)
        else:
            text = "<b>📈 Аномальні обсяги торгів:</b>\n\n" + "\n".join(anomalies[:8])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Error in volume_anomaly: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['advanced_analysis'])
def advanced_analysis_handler(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /advanced_analysis BTC")
            return
            
        symbol = parts[1].upper() + "USDT"
        msg = bot.send_message(message.chat.id, f"🔍 Аналізую {symbol}...")
        
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df.get("c", [])) < 50:
            bot.edit_message_text("❌ Не вдалося отримати дані для цього токена", message.chat.id, msg.message_id)
            return
        
        closes = [float(c) for c in df["c"]]
        volumes = [float(v) for v in df["v"]]
        last_price = closes[-1]
        
        pump_type, price_change, pump_data = detect_pump_activity(symbol, closes, volumes, USER_SETTINGS)
        is_volume_anomaly, volume_data = detect_volume_anomaly(symbol, volumes, USER_SETTINGS)
        volume_metrics = analyze_volume(volumes, USER_SETTINGS)
        
        report_text = f"<b>📊 Розширений аналіз {symbol}</b>\n\n"
        report_text += f"💰 Поточна ціна: ${last_price:.4f}\n"
        report_text += f"📈 Зміна за 24г: {price_change:+.1f}%\n"
        
        if pump_type:
            report_text += f"🚨 Тип події: {pump_type}\n"
            report_text += f"⚠️ Рівень ризику: {pump_data.get('risk_level', 5)}/10\n"
        
        report_text += f"📊 Волатильність: {calculate_volatility(closes[-24:]):.1f}%\n"
        report_text += f"💹 Співвідношення обсягу: {volume_metrics.get('volume_ratio', 0):.1f}x\n"
        
        if is_volume_anomaly:
            report_text += "🔴 Виявлено аномалію обсягу!\n"
        
        if pump_type == "PUMP" and pump_data.get('risk_level', 5) > 7:
            report_text += "\n🔻 Рекомендація: Високий ризик! Уникайте входу.\n"
        elif pump_type == "PUMP":
            report_text += "\n🟡 Рекомендація: Обережно! Можлива корекція.\n"
        elif price_change < -10:
            report_text += "\n🟢 Рекомендація: Можливий відскок після падіння.\n"
        else:
            report_text += "\n⚪ Рекомендація: Стандартна ситуація.\n"
        
        bot.edit_message_text(report_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in advanced_analysis: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    try:
        msg = bot.send_message(message.chat.id, "🔍 Аналізую ринок...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > USER_SETTINGS['min_volume']
        ]

        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:USER_SETTINGS['top_symbols']]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = [float(c) for c in df["c"]]
                volumes = [float(v) for v in df["v"]]
                last_price = closes[-1]

                rsi, vol_spike = calculate_technical_indicators(closes, volumes)
                sr_levels = find_support_resistance(
                    closes, 
                    window=USER_SETTINGS['window_size'], 
                    delta=USER_SETTINGS['sensitivity']
                )

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100

                    if last_price > lvl * 1.01 and diff_pct > 1:
                        signal = (
                            f"🚀 LONG breakout\n"
                            f"Пробито опір: ${lvl:.4f}\n"
                            f"Поточна ціна: ${last_price:.4f}\n"
                            f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                        )
                        break
                    elif last_price < lvl * 0.99 and diff_pct < -1:
                        signal = (
                            f"⚡ SHORT breakout\n"
                            f"Пробито підтримку: ${lvl:.4f}\n"
                            f"Поточна ціна: ${last_price:.4f}\n"
                            f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                        )
                        break

                event_type, price_change = detect_pump_dump(closes, volumes)
                
                if event_type:
                    signal = (
                        f"🔴 {event_type} DETECTED!\n"
                        f"Зміна ціни: {price_change:+.1f}%\n"
                        f"Рекомендація: {'Шорт' if event_type == 'PUMP' else 'Лонг'}\n"
                        f"RSI: {rsi:.1f} | Volume: {'📈' if vol_spike else '📉'}"
                    )

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}\n" + "-"*40)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not signals:
            bot.edit_message_text("ℹ️ Жодних сигналів не знайдено.", message.chat.id, msg.message_id)
        else:
            text = f"<b>📊 Smart Auto Signals</b>\n\n" + "\n".join(signals[:10])
            bot.edit_message_text(text, message.chat.id, msg.message_id, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error in smart_auto: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['check_token'])
def check_token_handler(message):
    try:
        symbol = message.text.split()[1].upper() + "USDT"
        df = get_klines(symbol, interval="1h", limit=200)
        
        if not df:
            bot.send_message(message.chat.id, "❌ Токен не знайдено або помилка даних")
            return
            
        closes = [float(c) for c in df["c"]]
        volumes = [float(v) for v in df["v"]]
        last_price = closes[-1]
        
        rsi, vol_spike = calculate_technical_indicators(closes, volumes)
        sr_levels = find_support_resistance(closes)
        event_type, price_change = detect_pump_dump(closes, volumes)
        
        analysis_text = f"""
<b>{symbol} Analysis</b>

Поточна ціна: ${last_price:.4f}
RSI: {rsi:.1f} {'(перекупленість)' if rsi > 70 else '(перепроданість)' if rsi < 30 else ''}
Обсяг: {'підвищений' if vol_spike else 'нормальний'}
Подія: {event_type if event_type else 'немає'} ({price_change:+.1f}%)

<b>Key Levels:</b>
"""
        for level in sr_levels[-5:]:
            distance_pct = (last_price - level) / level * 100
            analysis_text += f"{level:.4f} ({distance_pct:+.1f}%)\n"

        if event_type == "PUMP":
            analysis_text += "\n🔴 Рекомендація: Шорт (можливий корекція після пампу)"
        elif event_type == "DUMP":
            analysis_text += "\n🟢 Рекомендація: Лонг (можливий відскок після дампу)"

        bot.send_message(message.chat.id, analysis_text, parse_mode="HTML")
        
    except IndexError:
        bot.send_message(message.chat.id, "ℹ️ Використання: /check_token BTC")
    except Exception as e:
        logger.error(f"Error in check_token: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['stats'])
def market_stats(message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()
        
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 1000000]
        
        gainers = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']), reverse=True)[:5]
        losers = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']))[:5]
        
        stats_text = "<b>📈 Market Statistics</b>\n\n"
        stats_text += "<b>Top Gainers:</b>\n"
        for i, coin in enumerate(gainers, 1):
            stats_text += f"{i}. {coin['symbol']} +{float(coin['priceChangePercent']):.1f}%\n"
        
        stats_text += "\n<b>Top Losers:</b>\n"
        for i, coin in enumerate(losers, 1):
            stats_text += f"{i}. {coin['symbol']} {float(coin['priceChangePercent']):.1f}%\n"
            
        bot.send_message(message.chat.id, stats_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in stats: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['settings'])
def show_settings(message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(
        KeyboardButton("Мін. обсяг 📊"),
        KeyboardButton("Кількість монет 🔢"),
        KeyboardButton("Чутливість ⚖️"),
        KeyboardButton("PUMP % 📈"),
        KeyboardButton("DUMP % 📉"),
        KeyboardButton("Головне меню 🏠")
    )
    
    settings_text = f"""
Поточні налаштування:

Мінімальний обсяг: {USER_SETTINGS['min_volume']:,.0f} USDT
Кількість монет для аналізу: {USER_SETTINGS['top_symbols']}
Чутливість: {USER_SETTINGS['sensitivity'] * 100}%
PUMP поріг: {USER_SETTINGS['pump_threshold']}%
DUMP поріг: {USER_SETTINGS['dump_threshold']}%
"""
    bot.send_message(message.chat.id, settings_text, reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text == "Мін. обсяг 📊")
def set_min_volume(message):
    msg = bot.send_message(message.chat.id, "Введіть мінімальний обсяг торгів (USDT):")
    bot.register_next_step_handler(msg, process_min_volume)

def process_min_volume(message):
    try:
        volume = float(message.text.replace(',', '').replace(' ', ''))
        USER_SETTINGS['min_volume'] = volume
        bot.send_message(message.chat.id, f"Мінімальний обсяг встановлено: {volume:,.0f} USDT")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "Кількість монет 🔢")
def set_top_symbols(message):
    msg = bot.send_message(message.chat.id, "Введіть кількість монет для аналізу:")
    bot.register_next_step_handler(msg, process_top_symbols)

def process_top_symbols(message):
    try:
        count = int(message.text)
        USER_SETTINGS['top_symbols'] = count
        bot.send_message(message.chat.id, f"Кількість монет для аналізу встановлено: {count}")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть ціле число.")

@bot.message_handler(func=lambda message: message.text == "Чутливість ⚖️")
def set_sensitivity(message):
    msg = bot.send_message(message.chat.id, "Введіть чутливість (0.1-5.0%):")
    bot.register_next_step_handler(msg, process_sensitivity)

def process_sensitivity(message):
    try:
        sensitivity = float(message.text)
        if 0.1 <= sensitivity <= 5.0:
            USER_SETTINGS['sensitivity'] = sensitivity / 100
            bot.send_message(message.chat.id, f"Чутливість встановлено: {sensitivity}%")
        else:
            bot.send_message(message.chat.id, "❌ Значення повинно бути між 0.1 та 5.0")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "PUMP % 📈")
def set_pump_threshold(message):
    msg = bot.send_message(message.chat.id, "Введіть поріг для виявлення PUMP (%):")
    bot.register_next_step_handler(msg, process_pump_threshold)

def process_pump_threshold(message):
    try:
        threshold = float(message.text)
        USER_SETTINGS['pump_threshold'] = threshold
        bot.send_message(message.chat.id, f"PUMP поріг встановлено: {threshold}%")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "DUMP % 📉")
def set_dump_threshold(message):
    msg = bot.send_message(message.chat.id, "Введіть поріг для виявлення DUMP (%):")
    bot.register_next_step_handler(msg, process_dump_threshold)

def process_dump_threshold(message):
    try:
        threshold = float(message.text)
        USER_SETTINGS['dump_threshold'] = threshold
        bot.send_message(message.chat.id, f"DUMP поріг встановлено: {threshold}%")
    except ValueError:
        bot.send_message(message.chat.id, "❌ Неправильний формат. Введіть число.")

@bot.message_handler(func=lambda message: message.text == "Головне меню 🏠")
def main_menu(message):
    send_welcome(message)

@bot.message_handler(commands=['arbitrage'])
def arbitrage_handler(message):
    try:
        msg = bot.send_message(message.chat.id, "🔍 Шукаю арбітражні можливості...")
        
        prices = arbitrage_analyzer.get_ticker_prices()
        if not prices:
            bot.edit_message_text("❌ Не вдалося отримати дані з Binance", message.chat.id, msg.message_id)
            return
        
        opportunities = arbitrage_analyzer.find_triangular_arbitrage_pairs(prices)
        
        if not opportunities:
            bot.edit_message_text("ℹ️ Арбітражних можливостей не знайдено.", message.chat.id, msg.message_id)
            return
        
        message_text = "<b>🔎 Знайдені арбітражні можливості:</b>\n\n"
        
        for i, opportunity in enumerate(opportunities[:5]):
            message_text += f"{i+1}. {arbitrage_analyzer.format_opportunity_message(opportunity)}\n"
            message_text += "─" * 40 + "\n"
        
        bot.edit_message_text(message_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in arbitrage: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['market_depth'])
def market_depth_handler(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /market_depth BTCUSDT")
            return
            
        symbol = parts[1].upper()
        msg = bot.send_message(message.chat.id, f"🔍 Аналізую глибину ринку для {symbol}...")
        
        depth_analysis = arbitrage_analyzer.calculate_depth_arbitrage(symbol)
        
        if not depth_analysis:
            bot.edit_message_text("❌ Не вдалося проаналізувати глибину ринку", message.chat.id, msg.message_id)
            return
        
        report_text = f"<b>📊 Аналіз глибини ринку {symbol}</b>\n\n"
        report_text += f"Найкраща ціна купівлі: {depth_analysis['best_bid']:.8f}\n"
        report_text += f"Найкраща ціна продажу: {depth_analysis['best_ask']:.8f}\n"
        report_text += f"Спред: {depth_analysis['spread']:.8f}\n"
        report_text += f"Спред (%): {depth_analysis['spread_percentage']:.4f}%\n"
        report_text += f"Обсяг купівлі (топ-5): {depth_analysis['bid_volume']:.4f}\n"
        report_text += f"Обсяг продажу (топ-5): {depth_analysis['ask_volume']:.4f}\n"
        report_text += f"Диспропорція: {depth_analysis['imbalance']:.4f}\n\n"
        
        if depth_analysis['spread_percentage'] < 0.1:
            report_text += "🟢 Низький спред - хороша ліквідність\n"
        elif depth_analysis['spread_percentage'] < 0.5:
            report_text += "🟡 Середній спред - помірна ліквідність\n"
        else:
            report_text += "🔴 Високий спред - низька ліквідність\n"
            
        if depth_analysis['imbalance'] > 2:
            report_text += "📈 Сильний дисбаланс у бік купівлі\n"
        elif depth_analysis['imbalance'] < 0.5:
            report_text += "📉 Сильний дисбаланс у бік продажу\n"
        else:
            report_text += "⚖️ Збалансований ринок\n"
        
        bot.edit_message_text(report_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in market_depth: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['trade_signal'])
def trade_signal_handler(message):
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /trade_signal BTCUSDT")
            return
            
        symbol = parts[1].upper()
        msg = bot.send_message(message.chat.id, f"📊 Аналізую {symbol} для торгових сигналів...")
        
        signal = trade_assistant.generate_trade_signal(symbol)
        
        if 'error' in signal:
            bot.edit_message_text(f"❌ {signal['error']}", message.chat.id, msg.message_id)
            return
        
        response = f"🎯 <b>Торговий сигнал для {symbol}</b>\n\n"
        response += f"📈 Рекомендація: <b>{signal['recommendation']}</b>\n"
        response += f"💪 Впевненість: {signal['confidence']}%\n"
        response += f"⚠️ Рівень ризику: {signal['risk_level']}\n\n"
        
        response += "🎯 <b>Точки входу:</b>\n"
        for i, point in enumerate(signal['entry_points'], 1):
            response += f"{i}. ${point:.4f}\n"
        
        response += "\n🎯 <b>Точки виходу:</b>\n"
        for i, point in enumerate(signal['exit_points'], 1):
            response += f"{i}. ${point:.4f}\n"
        
        response += f"\n🕒 Оновлено: {signal['timestamp']}"
        
        bot.edit_message_text(response, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in trade_signal: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['whale_alert'])
def whale_alert_handler(message):
    try:
        msg = bot.send_message(message.chat.id, "🐋 Сканую активність китів...")
        
        alerts = whale_tracker.monitor_top_cryptos()
        
        if not alerts:
            bot.edit_message_text("ℹ️ Китової активності не виявлено", message.chat.id, msg.message_id)
            return
        
        message_text = "<b>🚨 АКТИВНІСТЬ КИТІВ:</b>\n\n"
        
        for i, alert in enumerate(alerts[:5]):
            message_text += f"{i+1}. {whale_tracker.format_whale_alert(alert)}\n"
            message_text += "─" * 40 + "\n"
        
        bot.edit_message_text(message_text, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in whale_alert: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

if __name__ == "__main__":
    bot.remove_webhook()
    
    def run_bot():
        logger.info("Запуск бота в режимі polling...")
        while True:
            try:
                bot.polling(none_stop=True, interval=3, timeout=20)
            except Exception as e:
                logger.error(f"Помилка бота: {e}")
                logger.info("Перезапуск бота через 10 секунд...")
                time.sleep(10)
    
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    
    @app.route('/health')
    def health():
        return "OK"
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
