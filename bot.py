from polygon import RESTClient
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple
import os
import requests
import logging
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from apscheduler.schedulers.background import BackgroundScheduler
import json
import hmac
import hashlib
import asyncio
import aiohttp
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Безпечне завантаження конфігурації
class Config:
    def __init__(self):
        self.POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "your_polygon_key_here")
        self.BOT_TOKEN = os.environ.get('BOT_TOKEN')
        self.BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
        self.BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
        
        if not self.BOT_TOKEN:
            logger.error("BOT_TOKEN не знайдено в змінних оточення")
            raise ValueError("BOT_TOKEN обов'язковий")

config = Config()
client = RESTClient(api_key=config.POLYGON_API_KEY)
bot = telebot.TeleBot(config.BOT_TOKEN)
app = Flask(__name__)

# ==================== ENHANCED TRADE ASSISTANT ====================
class EnhancedTradeAssistant:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.risk_free_rate = 0.02  # 2% річних
        self.volatility_lookback = 20
        
    async def get_market_data_async(self, symbol: str) -> Optional[Dict]:
        """Асинхронне отримання даних"""
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.get_klines_async(session, symbol, "1h", 100),
                    self.get_ticker_24hr_async(session, symbol),
                    self.get_depth_async(session, symbol)
                ]
                klines, ticker, depth = await asyncio.gather(*tasks)
                
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

    async def get_klines_async(self, session, symbol: str, interval: str, limit: int):
        try:
            url = f"{self.base_url}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            async with session.get(url, params=params, timeout=10) as response:
                return await response.json()
        except:
            return None

    async def get_ticker_24hr_async(self, session, symbol: str):
        try:
            url = f"{self.base_url}/ticker/24hr?symbol={symbol}"
            async with session.get(url, timeout=10) as response:
                return await response.json()
        except:
            return None

    async def get_depth_async(self, session, symbol: str):
        try:
            url = f"{self.base_url}/depth?symbol={symbol}&limit=20"
            async with session.get(url, timeout=10) as response:
                return await response.json()
        except:
            return None

    def calculate_advanced_indicators(self, closes: List[float]) -> Dict:
        """Розширені технічні індикатори"""
        if len(closes) < 20:
            return {}
            
        df = pd.DataFrame(closes, columns=['close'])
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = df['close'].rolling(14).min()
        high_14 = df['close'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ATR (Average True Range)
        high_low = df['close'].diff().abs()
        high_close = (df['close'] - df['close'].shift()).abs()
        low_close = (df['close'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df.iloc[-1].to_dict()

    def calculate_risk_metrics(self, closes: List[float]) -> Dict:
        """Метрики ризику"""
        returns = pd.Series(closes).pct_change().dropna()
        
        if len(returns) < 2:
            return {}
            
        volatility = returns.std() * np.sqrt(365)  # Річна волатильність
        sharpe = (returns.mean() * 365 - self.risk_free_rate) / volatility if volatility > 0 else 0
        max_drawdown = (pd.Series(closes) / pd.Series(closes).cummax() - 1).min()
        var_95 = returns.quantile(0.05)
        
        return {
            'volatility_annual': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'calmar_ratio': abs(returns.mean() * 365 / max_drawdown) if max_drawdown != 0 else 0
        }

    def generate_enhanced_signal(self, symbol: str) -> Dict:
        """Покращена генерація сигналів"""
        try:
            # Спрощений синхронний варіант для прикладу
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {'error': 'Could not fetch market data'}
            
            closes = [float(k[4]) for k in market_data['klines']]
            
            if len(closes) < 50:
                return {'error': 'Insufficient data'}
            
            # Базовий аналіз
            trend_analysis = self.analyze_trend(closes)
            volume_analysis = self.analyze_volume(market_data['klines'])
            momentum_analysis = self.analyze_momentum(closes)
            
            # Розширений аналіз
            advanced_indicators = self.calculate_advanced_indicators(closes)
            risk_metrics = self.calculate_risk_metrics(closes)
            
            # Комбінована рекомендація
            recommendation = self.generate_enhanced_recommendation(
                trend_analysis, volume_analysis, momentum_analysis, advanced_indicators, risk_metrics
            )
            
            return {
                'symbol': symbol,
                'recommendation': recommendation['action'],
                'confidence': recommendation['confidence'],
                'risk_level': recommendation['risk_level'],
                'entry_points': self.calculate_smart_entry_points(closes, advanced_indicators),
                'exit_points': self.calculate_smart_exit_points(closes, advanced_indicators),
                'advanced_indicators': advanced_indicators,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced signal generation: {e}")
            return {'error': str(e)}

    def generate_enhanced_recommendation(self, trend, volume, momentum, advanced, risk_metrics) -> Dict:
        """Покращена система рекомендацій"""
        score = 0
        factors = []
        
        # Тренд (30%)
        if trend['direction'] == 'up':
            score += 30 * min(trend['strength'] / 50, 1)  # Нормалізація
            factors.append(f"📈 Верхній тренд ({trend['strength']:.1f}%)")
        
        # Моментум (25%)
        if momentum['rsi'] < 30:
            score += 25
            factors.append("🔻 Перепроданість (RSI < 30)")
        elif momentum['rsi'] > 70:
            score -= 25
            factors.append("🔺 Перекупленість (RSI > 70)")
        
        # Об'єм (20%)
        if volume['volume_ratio'] > 2:
            score += 20
            factors.append(f"💨 Високий обсяг (x{volume['volume_ratio']:.1f})")
        
        # MACD (15%)
        if advanced.get('macd', 0) > advanced.get('macd_signal', 0):
            score += 15
            factors.append("📊 MACD позитивний")
        
        # Волатильність (10%)
        if risk_metrics.get('volatility_annual', 0) < 0.8:  # 80% річна волатильність
            score += 10
            factors.append("⚡ Низька волатильність")
        
        # Визначення дії
        if score >= 70:
            action = "STRONG_BUY"
            risk_level = "LOW"
        elif score >= 50:
            action = "BUY" 
            risk_level = "MEDIUM"
        elif score >= 30:
            action = "HOLD"
            risk_level = "MEDIUM"
        elif score >= 10:
            action = "SELL"
            risk_level = "HIGH"
        else:
            action = "STRONG_SELL"
            risk_level = "VERY_HIGH"
        
        return {
            'action': action,
            'confidence': min(95, max(5, score)),
            'risk_level': risk_level,
            'factors': factors,
            'score': score
        }

    def calculate_smart_entry_points(self, closes: List[float], indicators: Dict) -> List[float]:
        """Розумні точки входу на основі технічних рівнів"""
        current_price = closes[-1]
        
        # Використання Bollinger Bands для точок входу
        bb_lower = indicators.get('bb_lower', current_price * 0.95)
        bb_middle = indicators.get('bb_middle', current_price * 0.98)
        
        return [
            bb_lower,
            (bb_lower + bb_middle) / 2,
            bb_middle
        ]

    def calculate_smart_exit_points(self, closes: List[float], indicators: Dict) -> List[float]:
        """Розумні точки виходу"""
        current_price = closes[-1]
        
        bb_upper = indicators.get('bb_upper', current_price * 1.05)
        bb_middle = indicators.get('bb_middle', current_price * 1.02)
        
        return [
            bb_middle,
            (bb_middle + bb_upper) / 2,
            bb_upper
        ]

    # Збережемо оригінальні методи для сумісності
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

    def analyze_trend(self, closes):
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

    def analyze_momentum(self, closes):
        rsi = self.calculate_rsi(closes)
        
        return {
            'rsi': rsi,
            'momentum': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
            'price_acceleration': self.calculate_acceleration(closes)
        }

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

# ==================== PORTFOLIO MANAGER ====================
class PortfolioManager:
    def __init__(self):
        self.portfolio = {}
        self.risk_per_trade = 0.02  # 2% ризик на угоду
        
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """Розрахунок розміру позиції з управлінням ризиком"""
        risk_amount = account_balance * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
            
        position_size = risk_amount / price_diff
        return min(position_size, account_balance * 0.1)  # Макс 10% балансу

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Розрахунок співвідношення ризик/прибуток"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        return reward / risk if risk > 0 else 0

# ==================== ENHANCED WHALE TRACKER ====================
class EnhancedWhaleTracker:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.whale_threshold = 500000
        self.suspicious_patterns = []
        
    def detect_wash_trading(self, symbol: str) -> Optional[Dict]:
        """Виявлення мийних торгів (wash trading)"""
        try:
            trades = self.get_large_trades(symbol, 1000)
            if not trades:
                return None
                
            # Аналіз шаблонів торгів
            same_size_trades = {}
            for trade in trades:
                key = (trade['price'], trade['quantity'])
                same_size_trades[key] = same_size_trades.get(key, 0) + 1
                
            # Пошук підозрілих повторень
            suspicious = {k: v for k, v in same_size_trades.items() if v > 3}
            
            if suspicious:
                return {
                    'symbol': symbol,
                    'type': 'WASH_TRADING_SUSPECTED',
                    'suspicious_patterns': len(suspicious),
                    'details': list(suspicious.items())[:5]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error detecting wash trading: {e}")
            return None

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
    'rsi_oversold': 30,
    'risk_level': 'MEDIUM',
    'max_position_size': 0.1,
    'stop_loss_default': 0.03
}

ALERT_SUBSCRIPTIONS = {}
USER_PORTFOLIOS = {}
enhanced_trade_assistant = EnhancedTradeAssistant()
portfolio_manager = PortfolioManager()
enhanced_whale_tracker = EnhancedWhaleTracker()

# ==================== ENHANCED HELPER FUNCTIONS ====================
def safe_api_call(func):
    """Декоратор для безпечних API викликів"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper

@safe_api_call
def get_enhanced_klines(symbol, interval="1h", limit=200):
    """Покращене отримання даних з кешуванням"""
    cache_key = f"{symbol}_{interval}_{limit}"
    cache_time = 60  # секунд
    
    # Проста імітація кешу (в продакшені використовуйте Redis)
    if hasattr(get_enhanced_klines, 'cache'):
        cached_data, timestamp = get_enhanced_klines.cache.get(cache_key, (None, 0))
        if time.time() - timestamp < cache_time:
            return cached_data
    
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    
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
    
    # Зберігаємо в кеш
    if not hasattr(get_enhanced_klines, 'cache'):
        get_enhanced_klines.cache = {}
    get_enhanced_klines.cache[cache_key] = (df, time.time())
    
    return df

def calculate_multitimeframe_rsi(closes_1h, closes_4h, closes_1d):
    """RSI на кількох таймфреймах"""
    rsi_1h = calculate_rsi(closes_1h)
    rsi_4h = calculate_rsi(closes_4h) if len(closes_4h) >= 14 else 50
    rsi_1d = calculate_rsi(closes_1d) if len(closes_1d) >= 14 else 50
    
    return {
        '1h': rsi_1h,
        '4h': rsi_4h,
        '1d': rsi_1d,
        'average': (rsi_1h + rsi_4h + rsi_1d) / 3,
        'bullish_alignment': rsi_1h > rsi_4h > rsi_1d and all(rsi < 70 for rsi in [rsi_1h, rsi_4h, rsi_1d]),
        'bearish_alignment': rsi_1h < rsi_4h < rsi_1d and all(rsi > 30 for rsi in [rsi_1h, rsi_4h, rsi_1d])
    }

# ==================== ENHANCED BOT COMMANDS ====================

@bot.message_handler(commands=['start', 'help'])
def send_enhanced_welcome(message):
    """Покращена довідка"""
    help_text = """
🤖 <b>Enhanced Crypto Trading Bot</b>

🎯 <b>ОСНОВНІ КОМАНДИ:</b>
/analyze TICKER - Поглиблений аналіз токена
/smart_signal TICKER - Розширений торговий сигнал
/portfolio - Керування портфелем
/risk_check TICKER - Аналіз ризиків

📊 <b>СКАНЕРИ:</b>
/pump_scanner - Памп-можливості
/drop_scanner - Шорт-можливості  
/volume_breakout - Прогноз пробоїв
/market_health - Стан ринку

🐋 <b>АНАЛІТИКА:</b>
/whale_alert - Активність китів
/dark_pool - Аналіз темних пулів
/chain_reaction - Ланцюгові реакції

⚙️ <b>НАЛАШТУВАННЯ:</b>
/settings - Налаштування
/risk_settings - Управління ризиками

💡 <b>НОВИЙ ФУНКЦІОНАЛ:</b>
• AI-підсилена аналітика
• Багатотаймфреймний аналіз
• Керування ризиками
• Портфельний менеджер
"""
    
    keyboard = InlineKeyboardMarkup()
    keyboard.add(
        InlineKeyboardButton("📊 Аналіз токена", callback_data="analyze"),
        InlineKeyboardButton("🎯 Торгові сигнали", callback_data="signals")
    )
    keyboard.add(
        InlineKeyboardButton("🐋 Активність китів", callback_data="whale"),
        InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")
    )
    
    bot.send_message(message.chat.id, help_text, parse_mode="HTML", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    """Обробка callback-ів"""
    if call.data == "analyze":
        msg = bot.send_message(call.message.chat.id, "Введіть тикер токена (наприклад, BTC):")
        bot.register_next_step_handler(msg, process_analyze_ticker)
    elif call.data == "signals":
        show_smart_signals(call.message)
    elif call.data == "whale":
        enhanced_whale_alert_handler(call.message)
    elif call.data == "settings":
        show_enhanced_settings(call.message)

def process_analyze_ticker(message):
    """Обробка аналізу токена"""
    try:
        symbol = message.text.upper() + "USDT"
        msg = bot.send_message(message.chat.id, f"🔍 Детальний аналіз {symbol}...")
        
        # Отримуємо дані з різних таймфреймів
        klines_1h = get_enhanced_klines(symbol, "1h", 100)
        klines_4h = get_enhanced_klines(symbol, "4h", 100)
        klines_1d = get_enhanced_klines(symbol, "1d", 100)
        
        if not klines_1h:
            bot.edit_message_text("❌ Не вдалося отримати дані", message.chat.id, msg.message_id)
            return
        
        closes_1h = [float(c) for c in klines_1h["c"]]
        closes_4h = [float(c) for c in klines_4h["c"]] if klines_4h else closes_1h
        closes_1d = [float(c) for c in klines_1d["c"]] if klines_1d else closes_1h
        
        # Багатотаймфреймний аналіз
        multi_tf_rsi = calculate_multitimeframe_rsi(closes_1h, closes_4h, closes_1d)
        
        # Генерація сигналу
        signal = enhanced_trade_assistant.generate_enhanced_signal(symbol)
        
        if 'error' in signal:
            bot.edit_message_text(f"❌ {signal['error']}", message.chat.id, msg.message_id)
            return
        
        # Формування звіту
        report = generate_enhanced_analysis_report(symbol, signal, multi_tf_rsi)
        bot.edit_message_text(report, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in analyze: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

def generate_enhanced_analysis_report(symbol: str, signal: Dict, multi_tf_rsi: Dict) -> str:
    """Генерація покращеного звіту аналізу"""
    report = f"🎯 <b>ДЕТАЛЬНИЙ АНАЛІЗ {symbol}</b>\n\n"
    
    # Основна інформація
    report += f"📊 <b>РЕКОМЕНДАЦІЯ:</b> {signal['recommendation']}\n"
    report += f"💪 <b>ВПЕВНЕНІСТЬ:</b> {signal['confidence']}%\n"
    report += f"⚠️ <b>РИЗИК:</b> {signal['risk_level']}\n\n"
    
    # Багатотаймфреймний RSI
    report += f"📈 <b>RSI АНАЛІЗ:</b>\n"
    report += f"• 1 година: {multi_tf_rsi['1h']:.1f}\n"
    report += f"• 4 години: {multi_tf_rsi['4h']:.1f}\n"
    report += f"• 1 день: {multi_tf_rsi['1d']:.1f}\n"
    
    if multi_tf_rsi['bullish_alignment']:
        report += "• 🟢 Булліш вирівнювання!\n"
    elif multi_tf_rsi['bearish_alignment']:
        report += "• 🔴 Беаріш вирівнювання!\n"
    
    report += f"\n🎯 <b>ТОЧКИ ВХОДУ:</b>\n"
    for i, point in enumerate(signal['entry_points'], 1):
        report += f"{i}. ${point:.4f}\n"
    
    report += f"\n🎯 <b>ТОЧКИ ВИХОДУ:</b>\n"
    for i, point in enumerate(signal['exit_points'], 1):
        report += f"{i}. ${point:.4f}\n"
    
    # Метрики ризику
    if 'risk_metrics' in signal:
        rm = signal['risk_metrics']
        report += f"\n⚡ <b>МЕТРИКИ РИЗИКУ:</b>\n"
        report += f"• Sharpe Ratio: {rm.get('sharpe_ratio', 0):.2f}\n"
        report += f"• Макс. просідання: {rm.get('max_drawdown', 0)*100:.1f}%\n"
        report += f"• VaR (95%): {rm.get('var_95', 0)*100:.1f}%\n"
    
    report += f"\n🕒 <b>ОНОВЛЕНО:</b> {datetime.now().strftime('%H:%M:%S')}"
    
    return report

@bot.message_handler(commands=['smart_signal'])
def enhanced_signal_handler(message):
    """Покращена команда торгових сигналів"""
    try:
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "ℹ️ Використання: /smart_signal BTCUSDT")
            return
            
        symbol = parts[1].upper()
        msg = bot.send_message(message.chat.id, f"🤖 Генерація AI-сигналу для {symbol}...")
        
        signal = enhanced_trade_assistant.generate_enhanced_signal(symbol)
        
        if 'error' in signal:
            bot.edit_message_text(f"❌ {signal['error']}", message.chat.id, msg.message_id)
            return
        
        response = f"🎯 <b>AI ТОРГОВИЙ СИГНАЛ ДЛЯ {symbol}</b>\n\n"
        response += f"📊 Дія: <b>{signal['recommendation']}</b>\n"
        response += f"💪 Впевненість: {signal['confidence']}%\n"
        response += f"⚠️ Рівень ризику: {signal['risk_level']}\n"
        
        if 'factors' in signal:
            response += f"\n🔍 <b>ФАКТОРИ:</b>\n"
            for factor in signal['factors'][:5]:
                response += f"• {factor}\n"
        
        response += f"\n🎯 <b>ТОЧКИ ВХОДУ:</b>\n"
        for i, point in enumerate(signal['entry_points'], 1):
            response += f"{i}. ${point:.4f}\n"
        
        response += f"\n🎯 <b>ТОЧКИ ВИХОДУ:</b>\n"
        for i, point in enumerate(signal['exit_points'], 1):
            response += f"{i}. ${point:.4f}\n"
        
        # Розрахунок позиції
        if 'entry_points' in signal and signal['entry_points']:
            entry = signal['entry_points'][0]
            stop_loss = min(signal['entry_points']) * 0.97  # -3% stop loss
            take_profit = max(signal['exit_points'])
            
            risk_reward = portfolio_manager.calculate_risk_reward_ratio(entry, stop_loss, take_profit)
            response += f"\n⚖️ <b>РИЗИК/ПРИБУТОК:</b> 1:{risk_reward:.1f}\n"
        
        response += f"\n🕒 Оновлено: {signal['timestamp'][11:19]}"
        
        # Додаємо кнопки дій
        keyboard = InlineKeyboardMarkup()
        keyboard.add(
            InlineKeyboardButton("📊 Детальний аналіз", callback_data=f"analyze_{symbol}"),
            InlineKeyboardButton("⚡ Швидка угода", callback_data=f"trade_{symbol}")
        )
        
        bot.edit_message_text(response, message.chat.id, msg.message_id, parse_mode="HTML", reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Error in smart_signal: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

@bot.message_handler(commands=['market_health'])
def market_health_handler(message):
    """Аналіз загального стану ринку"""
    try:
        msg = bot.send_message(message.chat.id, "🏥 Аналізую здоров'я ринку...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=15).json()
        
        # Аналіз топ-20 монет за обсягом
        usdt_pairs = [d for d in data if isinstance(d, dict) and d.get("symbol", "").endswith("USDT")]
        top_symbols = sorted(usdt_pairs, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:20]
        
        health_metrics = calculate_market_health(top_symbols)
        
        report = generate_market_health_report(health_metrics)
        bot.edit_message_text(report, message.chat.id, msg.message_id, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in market_health: {e}")
        bot.send_message(message.chat.id, f"❌ Помилка: {e}")

def calculate_market_health(symbols_data: List[Dict]) -> Dict:
    """Розрахунок метрик здоров'я ринку"""
    price_changes = [float(s.get("priceChangePercent", 0)) for s in symbols_data]
    volumes = [float(s.get("quoteVolume", 0)) for s in symbols_data]
    
    avg_price_change = sum(price_changes) / len(price_changes)
    bullish_count = sum(1 for change in price_changes if change > 0)
    bearish_count = sum(1 for change in price_changes if change < 0)
    
    # Волатильність ринку
    volatility = np.std(price_changes) if price_changes else 0
    
    # Індекс страху та жадібності (спрощений)
    fear_greed = calculate_fear_greed_index(price_changes, volumes)
    
    return {
        'avg_price_change': avg_price_change,
        'bullish_ratio': bullish_count / len(symbols_data),
        'volatility': volatility,
        'fear_greed_index': fear_greed,
        'market_sentiment': 'BULLISH' if avg_price_change > 0 else 'BEARISH',
        'total_volume': sum(volumes)
    }

def calculate_fear_greed_index(price_changes: List[float], volumes: List[float]) -> int:
    """Спрощений розрахунок індексу страху та жадібності"""
    if not price_changes:
        return 50
        
    # Базується на середній зміні цін та обсягах
    avg_change = sum(price_changes) / len(price_changes)
    volume_trend = sum(volumes[-min(5, len(volumes)):]) / sum(volumes[-10:]) if len(volumes) >= 10 else 1
    
    base_score = 50
    
    # Корекція на основі ціни
    if avg_change > 5:
        base_score += 25
    elif avg_change > 2:
        base_score += 15
    elif avg_change < -5:
        base_score -= 25
    elif avg_change < -2:
        base_score -= 15
    
    # Корекція на основі обсягів
    if volume_trend > 1.2:
        base_score += 10
    elif volume_trend < 0.8:
        base_score -= 10
    
    return max(0, min(100, base_score))

def generate_market_health_report(metrics: Dict) -> str:
    """Генерація звіту про здоров'я ринку"""
    report = "🏥 <b>АНАЛІЗ ЗДОРОВ'Я РИНКУ</b>\n\n"
    
    # Загальний стан
    sentiment_emoji = "🟢" if metrics['market_sentiment'] == 'BULLISH' else "🔴"
    report += f"{sentiment_emoji} <b>ЗАГАЛЬНИЙ НАСТРІЙ:</b> {metrics['market_sentiment']}\n"
    report += f"📈 <b>Середня зміна:</b> {metrics['avg_price_change']:+.2f}%\n"
    report += f"📊 <b>Буллішних монет:</b> {metrics['bullish_ratio']*100:.1f}%\n\n"
    
    # Індекс страху та жадібності
    fgi = metrics['fear_greed_index']
    if fgi >= 75:
        fgi_status = "ЕКСТРЕМАЛЬНА ЖАДІБНІСТЬ 🤑"
    elif fgi >= 60:
        fgi_status = "ЖАДІБНІСТЬ 😊"
    elif fgi >= 40:
        fgi_status = "НЕЙТРАЛЬНИЙ 😐"
    elif fgi >= 25:
        fgi_status = "СТРАХ 😨"
    else:
        fgi_status = "ЕКСТРЕМАЛЬНИЙ СТРАХ 😱"
    
    report += f"🎭 <b>ІНДЕКС СТРАХУ/ЖАДІБНОСТІ:</b> {fgi}/100\n"
    report += f"📊 <b>СТАН:</b> {fgi_status}\n\n"
    
    # Волатильність
    volatility = metrics['volatility']
    if volatility > 5:
        vol_status = "ВИСОКА ⚠️"
    elif volatility > 2:
        vol_status = "ПОМІРНА 📊"
    else:
        vol_status = "НИЗЬКА ✅"
    
    report += f"⚡ <b>ВОЛАТИЛЬНІСТЬ:</b> {volatility:.2f}% ({vol_status})\n"
    report += f"💎 <b>ЗАГАЛЬНИЙ ОБСЯГ:</b> ${metrics['total_volume']:,.0f}\n\n"
    
    # Рекомендації
    report += "💡 <b>РЕКОМЕНДАЦІЇ:</b>\n"
    if metrics['bullish_ratio'] > 0.7 and fgi < 70:
        report += "• 📈 Сильний булліш тренд\n• 🟢 Можна додавати в лонги\n"
    elif metrics['bullish_ratio'] < 0.3 and fgi > 30:
        report += "• 📉 Сильний беаріш тренд\n• 🔴 Можливі шорт-можливості\n"
    else:
        report += "• ⚖️ Ринок у рівновазі\n• 📊 Чекайте чітких сигналів\n"
    
    report += f"\n🕒 Оновлено: {datetime.now().strftime('%H:%M:%S')}"
    
    return report

@bot.message_handler(commands=['risk_settings'])
def risk_settings_handler(message):
    """Налаштування управління ризиками"""
    keyboard = InlineKeyboardMarkup()
    keyboard.add(
        InlineKeyboardButton("📊 Змінити рівень ризику", callback_data="change_risk"),
        InlineKeyboardButton("💼 Макс. розмір позиції", callback_data="change_position_size"),
        InlineKeyboardButton("🛑 Стоп-лосс за замовчуванням", callback_data="change_stop_loss")
    )
    
    settings_text = f"""
⚙️ <b>НАЛАШТУВАННЯ РИЗИКІВ</b>

Поточні налаштування:
• 📊 Рівень ризику: {USER_SETTINGS['risk_level']}
• 💼 Макс. позиція: {USER_SETTINGS['max_position_size']*100}%
• 🛑 Стоп-лосс: {USER_SETTINGS['stop_loss_default']*100}%

💡 <b>Рекомендації:</b>
• Консервативний: 1-2% ризик на угоду
• Помірний: 2-3% ризик на угоду  
• Агресивний: 3-5% ризик на угоду
"""
    
    bot.send_message(message.chat.id, settings_text, parse_mode="HTML", reply_markup=keyboard)

# ==================== SCHEDULER & BACKGROUND TASKS ====================
scheduler = BackgroundScheduler()

def enhanced_alert_system():
    """Покращена система сповіщень"""
    if not ALERT_SUBSCRIPTIONS:
        return
    
    try:
        # Перевірка ринкових умов
        health_metrics = calculate_market_health(get_top_symbols())
        
        # Сповіщення про екстремальні умови
        if health_metrics['fear_greed_index'] >= 80:
            alert = "🚨 ЕКСТРЕМАЛЬНА ЖАДІБНІСТЬ! Можлива корекція."
            send_bulk_alert(alert)
        elif health_metrics['fear_greed_index'] <= 20:
            alert = "🚨 ЕКСТРЕМАЛЬНИЙ СТРАХ! Можливий відскок."
            send_bulk_alert(alert)
            
    except Exception as e:
        logger.error(f"Error in alert system: {e}")

def send_bulk_alert(alert_text: str):
    """Масове відправлення сповіщень"""
    for chat_id in ALERT_SUBSCRIPTIONS.keys():
        try:
            bot.send_message(chat_id, alert_text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Error sending alert to {chat_id}: {e}")

def get_top_symbols(limit: int = 50):
    """Отримання топових символів"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()
    
    symbols = [
        d for d in data
        if isinstance(d, dict) and
        d.get("symbol", "").endswith("USDT") and 
        float(d.get("quoteVolume", 0)) > USER_SETTINGS['min_volume']
    ]

    return sorted(
        symbols,
        key=lambda x: float(x.get("quoteVolume", 0)),
        reverse=True
    )[:limit]

# Додаємо завдання планувальника
scheduler.add_job(enhanced_alert_system, 'interval', minutes=30)
scheduler.add_job(send_alerts_to_subscribers, 'interval', minutes=15)

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return "Enhanced Crypto Bot is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook для зовнішніх сповіщень"""
    try:
        data = request.get_json()
        # Обробка webhook даних
        logger.info(f"Webhook received: {data}")
        return "OK"
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return "Error", 400

# ==================== ЗАПУСК СИСТЕМИ ====================
def initialize_system():
    """Ініціалізація системи"""
    logger.info("Запуск покращеної системи...")
    
    # Перевірка API ключів
    if config.POLYGON_API_KEY == "your_polygon_key_here":
        logger.warning("Polygon API ключ не налаштовано")
    
    # Запуск планувальника
    if not scheduler.running:
        scheduler.start()
        logger.info("Планувальник запущено")

def run_bot_safe():
    """Безпечний запуск бота"""
    logger.info("Запуск бота в режимі polling...")
    
    while True:
        try:
            bot.polling(none_stop=True, interval=3, timeout=20)
        except Exception as e:
            logger.error(f"Помилка бота: {e}")
            logger.info("Перезапуск бота через 10 секунд...")
            time.sleep(10)

if __name__ == "__main__":
    bot.remove_webhook()
    
    # Ініціалізація
    initialize_system()
    
    # Запуск в окремому потоці
    bot_thread = threading.Thread(target=run_bot_safe)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Запуск Flask додатку
    port = int(os.environ.get('PORT', 5000))
    
    @app.route('/health')
    def health():
        return json.dumps({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Додаємо оригінальні функції для сумісності
def calculate_rsi(prices, period=14):
    """RSI calculation for compatibility"""
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

def send_alerts_to_subscribers():
    """Original alert system for compatibility"""
    # Implementation from original code
    pass

def enhanced_whale_alert_handler(message):
    """Enhanced whale alert handler"""
    # Implementation would go here
    bot.send_message(message.chat.id, "🐋 Розширений моніторинг китів в розробці...")

def show_enhanced_settings(message):
    """Enhanced settings menu"""
    # Implementation would go here
    bot.send_message(message.chat.id, "⚙️ Розширені налаштування в розробці...")

def show_smart_signals(message):
    """Smart signals display"""
    # Implementation would go here  
    bot.send_message(message.chat.id, "🎯 Розумні сигнали в розробці...")