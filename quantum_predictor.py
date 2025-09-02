# quantum_predictor.py
import requests
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from utils import safe_request

logger = logging.getLogger(__name__)

class QuantumPredictor:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.quantum_state = {}
        self.schrodinger_cache = {}
        self.entanglement_matrix = {}
        self.last_analysis_time = {}
        
    def initialize_quantum_state(self):
        """Ініціалізація квантового стану ринку"""
        self.quantum_state = {
            'superposition': {},
            'entanglement': {},
            'probability_waves': {},
            'quantum_coherence': 0.95,
            'heisenberg_uncertainty': 0.12
        }
    
    def schrodinger_analysis(self, symbol: str) -> Dict[str, float]:
        """Аналіз кота Шредінгера для токена"""
        try:
            # Кешування результатів на 5 хвилин
            current_time = time.time()
            if symbol in self.schrodinger_cache:
                cached_data, cache_time = self.schrodinger_cache[symbol]
                if current_time - cache_time < 300:  # 5 хвилин
                    return cached_data
            
            klines = safe_request(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': '5m', 'limit': 100}
            )
            
            if not klines or not isinstance(klines, list):
                return {'alive': 0.5, 'dead': 0.5, 'superposition': 0.5, 'quantum_entropy': 0.5, 'wave_amplitude': 0.5}
            
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            if len(closes) < 10:
                return {'alive': 0.5, 'dead': 0.5, 'superposition': 0.5, 'quantum_entropy': 0.5, 'wave_amplitude': 0.5}
            
            wave_function = self.calculate_wave_function(closes)
            probability_density = self.calculate_probability_density(volumes)
            quantum_entropy = self.calculate_quantum_entropy(closes)
            
            # Покращений розрахунок ймовірності
            price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[:5]) if np.mean(volumes[:5]) != 0 else 1
            
            alive_probability = 0.5 + (wave_function * 0.2 + probability_density * 0.1 + 
                                     np.tanh(price_change * 10) * 0.1 + np.tanh(volume_trend - 1) * 0.1)
            
            alive_probability = max(0.1, min(0.9, alive_probability))
            dead_probability = 1 - alive_probability
            
            result = {
                'alive': alive_probability,
                'dead': dead_probability,
                'superposition': abs(alive_probability - dead_probability),
                'quantum_entropy': quantum_entropy,
                'wave_amplitude': wave_function
            }
            
            # Зберігаємо в кеш
            self.schrodinger_cache[symbol] = (result, current_time)
            return result
            
        except Exception as e:
            logger.error(f"Quantum error in schrodinger_analysis for {symbol}: {e}")
            return {'alive': 0.5, 'dead': 0.5, 'superposition': 0.5, 'quantum_entropy': 0.5, 'wave_amplitude': 0.5}
    
    def calculate_wave_function(self, prices: List[float]) -> float:
        """Розрахунок хвильової функції ціни"""
        if len(prices) < 10:
            return 0.5
            
        try:
            prices_array = np.array(prices)
            normalized_prices = (prices_array - np.min(prices_array)) / (np.max(prices_array) - np.min(prices_array) + 1e-10)
            
            # Аналіз Fourier
            fft = np.fft.fft(normalized_prices - np.mean(normalized_prices))
            dominant_freq = np.max(np.abs(fft[1:len(fft)//2]))  # Ігноруємо постійну складову
            
            return min(1.0, dominant_freq / len(prices) * 5)
        except:
            return 0.5
    
    def calculate_probability_density(self, volumes: List[float]) -> float:
        """Розрахунок щільності ймовірності за обсягами"""
        if len(volumes) < 10:
            return 0.5
            
        try:
            volume_array = np.array(volumes)
            mean_volume = np.mean(volume_array)
            if mean_volume == 0:
                return 0.5
                
            # Коефіцієнт варіації
            cv = np.std(volume_array) / mean_volume
            return min(1.0, cv * 1.5)
        except:
            return 0.5
    
    def calculate_quantum_entropy(self, prices: List[float]) -> float:
        """Розрахунок квантової ентропії"""
        if len(prices) < 10:
            return 0.5
            
        try:
            returns = np.diff(prices) / prices[:-1]
            if len(returns) < 5:
                return 0.5
                
            # Ентропія Шеннона
            hist, bin_edges = np.histogram(returns, bins=min(10, len(returns)//2), density=True)
            hist = hist[hist > 0]
            
            if len(hist) == 0:
                return 0.5
                
            entropy = -np.sum(hist * np.log2(hist))
            return min(1.0, entropy / np.log2(len(hist)))
        except:
            return 0.5
    
    def predict_quantum_jumps(self, symbols: List[str]) -> List[Dict]:
        """Прогнозування квантових стрибків"""
        predictions = []
        
        try:
            # Аналіз заплутаності для контексту
            entanglement_scores = self.quantum_entanglement_analysis(symbols)
            
            for symbol in symbols:
                schrodinger_state = self.schrodinger_analysis(symbol)
                current_price = self.get_current_price(symbol)
                
                if current_price <= 0:
                    continue
                
                # Покращений прогноз напрямку (менше випадковості)
                recent_trend = self.get_recent_trend(symbol)
                volume_analysis = self.analyze_volume(symbol)
                
                # Комбінуємо сигнали для кращого прогнозу
                if recent_trend > 0.1 and volume_analysis > 0.6:
                    jump_direction = 'UP'
                elif recent_trend < -0.1 and volume_analysis > 0.6:
                    jump_direction = 'DOWN'
                else:
                    # Якщо сигнали слабкі, використовуємо квантовий стан
                    jump_direction = 'UP' if schrodinger_state['alive'] > 0.6 else 'DOWN'
                
                jump_probability = (schrodinger_state['alive'] * 0.4 + 
                                  max(0, recent_trend) * 0.3 + 
                                  volume_analysis * 0.3)
                
                volatility = self.calculate_volatility(symbol)
                target_price = self.calculate_quantum_target(current_price, jump_direction, volatility)
                
                confidence = min(95, jump_probability * 100 * 0.85)
                
                if confidence > 60:
                    predictions.append({
                        'symbol': symbol,
                        'direction': jump_direction,
                        'confidence': confidence,
                        'current_price': current_price,
                        'target_price': target_price,
                        'probability_alive': schrodinger_state['alive'],
                        'quantum_entropy': schrodinger_state['quantum_entropy'],
                        'timeframe': self.quantum_timeframe(schrodinger_state),
                        'risk_level': self.calculate_quantum_risk(schrodinger_state)
                    })
            
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions[:8]  # Обмежуємо кількість прогнозів
            
        except Exception as e:
            logger.error(f"Quantum jump prediction error: {e}")
            return []
    
    def get_recent_trend(self, symbol: str) -> float:
        """Аналіз останнього тренду"""
        try:
            klines = safe_request(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': '15m', 'limit': 10}
            )
            
            if klines and isinstance(klines, list) and len(klines) >= 5:
                closes = [float(k[4]) for k in klines]
                recent_prices = closes[-5:]
                trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                return max(-0.3, min(0.3, trend))  # Обмежуємо тренд
            return 0
        except:
            return 0
    
    def analyze_volume(self, symbol: str) -> float:
        """Аналіз обсягів торгів"""
        try:
            klines = safe_request(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': '15m', 'limit': 20}
            )
            
            if klines and isinstance(klines, list) and len(klines) >= 10:
                volumes = [float(k[5]) for k in klines]
                recent_volumes = volumes[-5:]
                avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else np.mean(volumes)
                
                if avg_volume == 0:
                    return 0.5
                    
                volume_ratio = np.mean(recent_volumes) / avg_volume
                return min(1.0, volume_ratio / 2)
            return 0.5
        except:
            return 0.5
    
    def calculate_quantum_target(self, current_price: float, direction: str, volatility: float) -> float:
        """Розрахунок квантової цільової ціни"""
        # Більш реалістичний розрахунок цільової ціни
        move_pct = volatility * 0.6  # 60% від волатильності
        
        # Обмежуємо максимальний рух
        move_pct = min(10.0, move_pct)  # Не більше 10%
        
        if direction == 'UP':
            return current_price * (1 + move_pct / 100)
        else:
            return current_price * (1 - move_pct / 100)
    
    # Решта методів залишаються незмінними...
    def quantum_timeframe(self, schrodinger_state: Dict) -> str:
        """Визначення квантового таймфрейму"""
        entropy = schrodinger_state.get('quantum_entropy', 0.5)
        
        if entropy > 0.7:
            return "15-30 хвилин"
        elif entropy > 0.4:
            return "30-60 хвилин"
        else:
            return "1-2 години"
    
    def calculate_quantum_risk(self, schrodinger_state: Dict) -> str:
        """Розрахунок квантового ризику"""
        superposition = schrodinger_state.get('superposition', 0.5)
        
        if superposition > 0.7:
            return "ВИСОКИЙ ⚠️"
        elif superposition > 0.4:
            return "СЕРЕДНІЙ 🟡"
        else:
            return "НИЗЬКИЙ 🟢"
    
    def generate_quantum_strategy(self, prediction: Dict) -> str:
        """Генерація квантової стратегії"""
        symbol = prediction['symbol']
        direction = prediction['direction']
        
        entry_offset = 0.5 if prediction['risk_level'] == "ВИСОКИЙ ⚠️" else 0.3
        
        if direction == 'UP':
            entry_price = prediction['current_price'] * (1 + entry_offset/100)
            return (f"🟢 КВАНТОВА КУПІВЛЯ {symbol}\n"
                    f"⚡ Вхід: ${entry_price:.6f}\n"
                    f"🎯 Ціль: ${prediction['target_price']:.6f}\n"
                    f"⏰ Час: {prediction['timeframe']}\n"
                    f"📊 Впевненість: {prediction['confidence']:.1f}%")
        else:
            entry_price = prediction['current_price'] * (1 - entry_offset/100)
            return (f"🔴 КВАНТОВИЙ ШОРТ {symbol}\n"
                    f"⚡ Вхід: ${entry_price:.6f}\n"
                    f"🎯 Ціль: ${prediction['target_price']:.6f}\n"
                    f"⏰ Час: {prediction['timeframe']}\n"
                    f"📊 Впевненість: {prediction['confidence']:.1f}%")

# Глобальний екземпляр
quantum_predictor = QuantumPredictor()