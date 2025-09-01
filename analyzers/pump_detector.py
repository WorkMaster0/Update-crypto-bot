import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def detect_pump_activity(symbol, closes, volumes, settings):
    """Розширений детектор памп-активності"""
    if len(closes) < 24:
        return None, 0, {}
    
    # Основні метрики
    price_change_24h = (closes[-1] - closes[-24]) / closes[-24] * 100
    price_change_1h = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
    
    # Аналіз обсягів
    volume_metrics = analyze_volume(volumes, settings)
    
    # Додаткові показники
    volatility = calculate_volatility(closes[-24:])
    green_candles = count_green_candles(closes[-24:])
    
    # Визначення пампу
    is_pump = (
        price_change_24h > settings['pump_threshold'] and
        volume_metrics['volume_spike'] and
        price_change_1h > 5 and  # Різкий зліт за останню годину
        green_candles > 15  # Більшість свічок зростаючі
    )
    
    if not is_pump:
        return None, price_change_24h, volume_metrics
    
    # Рівень ризику (1-10)
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
    """Детальний аналіз обсягів торгів"""
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
    """Розрахунок волатильності"""
    if len(prices) < 2:
        return 0
    
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    return sum(abs(r) for r in returns) / len(returns) * 100

def count_green_candles(prices):
    """Підрахунок зростаючих свічок"""
    if len(prices) < 2:
        return 0
    
    green_count = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            green_count += 1
    
    return green_count

def calculate_pump_risk(closes, volumes, price_change):
    """Розрахунок рівня ризику пампу"""
    risk = 5  # Базовий рівень
    
    # Корекція на основі величини зростання
    if price_change > 50:
        risk += 3
    elif price_change > 30:
        risk += 2
    elif price_change > 15:
        risk += 1
    
    # Корекція на основі обсягів
    if len(volumes) > 0:
        volume_ratio = volumes[-1] / (sum(volumes[-10:]) / 10) if sum(volumes[-10:]) > 0 else 1
        if volume_ratio > 5:
            risk += 2
        elif volume_ratio > 3:
            risk += 1
    
    # Обмеження від 1 до 10
    return max(1, min(10, risk))