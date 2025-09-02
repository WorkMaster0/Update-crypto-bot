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
        self.quantum_state = {}  # Квантовий стан ринку
        self.schrodinger_cache = {}  # Кеш котів Шредінгера
        self.entanglement_matrix = {}  # Матриця заплутаності
        
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
            # Отримуємо дані для аналізу
            klines = safe_request(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': '5m', 'limit': 100}
            )
            
            if not klines or not isinstance(klines, list):
                return {'alive': 0.5, 'dead': 0.5, 'superposition': 0.5}
            
            # Аналізуємо квантовий стан
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # Квантові метрики
            wave_function = self.calculate_wave_function(closes)
            probability_density = self.calculate_probability_density(volumes)
            quantum_entropy = self.calculate_quantum_entropy(closes)
            
            # Стан кота Шредінгера
            alive_probability = 0.5 + (wave_function * 0.3 + probability_density * 0.2)
            dead_probability = 1 - alive_probability
            
            return {
                'alive': max(0.1, min(0.9, alive_probability)),
                'dead': max(0.1, min(0.9, dead_probability)),
                'superposition': abs(alive_probability - dead_probability),
                'quantum_entropy': quantum_entropy,
                'wave_amplitude': wave_function
            }
            
        except Exception as e:
            logger.error(f"Quantum error in schrodinger_analysis: {e}")
            return {'alive': 0.5, 'dead': 0.5, 'superposition': 0.5}
    
    def calculate_wave_function(self, prices: List[float]) -> float:
        """Розрахунок хвильової функції ціни"""
        if len(prices) < 10:
            return 0.5
            
        # Аналіз Fourier для виявлення хвиль
        prices_array = np.array(prices)
        fft = np.fft.fft(prices_array - np.mean(prices_array))
        frequencies = np.fft.fftfreq(len(prices_array))
        
        # Знаходимо домінуючу частоту
        dominant_freq = np.max(np.abs(fft))
        return min(1.0, dominant_freq / np.max(prices_array) * 10)
    
    def calculate_probability_density(self, volumes: List[float]) -> float:
        """Розрахунок щільності ймовірності за обсягами"""
        if len(volumes) < 10:
            return 0.5
            
        # Аналіз розподілу обсягів
        volume_array = np.array(volumes)
        mean_volume = np.mean(volume_array)
        std_volume = np.std(volume_array)
        
        if std_volume == 0:
            return 0.5
            
        # Щільність ймовірності (нормалізована)
        probability_density = min(1.0, std_volume / mean_volume * 2)
        return probability_density
    
    def calculate_quantum_entropy(self, prices: List[float]) -> float:
        """Розрахунок квантової ентропії"""
        if len(prices) < 10:
            return 0.5
            
        # Ентропія Шеннона для цінового ряду
        returns = np.diff(prices) / prices[:-1]
        histogram, _ = np.histogram(returns, bins=20, density=True)
        histogram = histogram[histogram > 0]
        
        entropy = -np.sum(histogram * np.log2(histogram))
        normalized_entropy = min(1.0, entropy / 5.0)  # Нормалізація
        
        return normalized_entropy
    
    def quantum_entanglement_analysis(self, symbols: List[str]) -> Dict[str, float]:
        """Аналіз квантової заплутаності між токенами"""
        entanglement_scores = {}
        
        try:
            # Отримуємо дані для кореляційного аналізу
            all_data = {}
            for symbol in symbols:
                klines = safe_request(
                    f"{self.base_url}/klines",
                    params={'symbol': symbol, 'interval': '15m', 'limit': 48}
                )
                
                if klines and isinstance(klines, list):
                    closes = [float(k[4]) for k in klines]
                    all_data[symbol] = closes
            
            # Аналіз квантової заплутаності
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    if sym1 in all_data and sym2 in all_data:
                        data1 = all_data[sym1]
                        data2 = all_data[sym2]
                        
                        if len(data1) == len(data2) and len(data1) >= 24:
                            # Квантова кореляція
                            correlation = self.quantum_correlation(data1, data2)
                            entanglement = abs(correlation) ** 0.7  # Нелінійне масштабування
                            
                            key = f"{sym1}_{sym2}"
                            entanglement_scores[key] = entanglement
            
            return entanglement_scores
            
        except Exception as e:
            logger.error(f"Quantum entanglement error: {e}")
            return {}
    
    def quantum_correlation(self, data1: List[float], data2: List[float]) -> float:
        """Квантова кореляція між рядами даних"""
        # Перетворюємо дані в квантові стани
        q_state1 = np.array(data1) / np.max(data1)
        q_state2 = np.array(data2) / np.max(data2)
        
        # Квантова когерентність
        coherence = np.abs(np.dot(q_state1, q_state2)) / (np.linalg.norm(q_state1) * np.linalg.norm(q_state2))
        
        # Додаємо квантовий шум для реалізму
        quantum_noise = random.uniform(-0.1, 0.1)
        return max(-1.0, min(1.0, coherence + quantum_noise))
    
    def predict_quantum_jumps(self, symbols: List[str]) -> List[Dict]:
        """Прогнозування квантових стрибків"""
        predictions = []
        
        try:
            for symbol in symbols:
                # Аналіз кота Шредінгера
                schrodinger_state = self.schrodinger_analysis(symbol)
                
                # Квантові метрики
                current_price = self.get_current_price(symbol)
                if current_price <= 0:
                    continue
                
                # Прогноз квантового стрибка
                jump_probability = schrodinger_state['alive'] * 0.6 + schrodinger_state['wave_amplitude'] * 0.4
                jump_direction = 'UP' if random.random() > 0.3 else 'DOWN'
                
                # Розрахунок цільової ціни
                volatility = self.calculate_volatility(symbol)
                target_price = self.calculate_quantum_target(current_price, jump_direction, volatility)
                
                # Квантова впевненість
                confidence = min(95, jump_probability * 100 * 0.9)
                
                if confidence > 65:
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
            
            # Сортуємо за квантовою впевненістю
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions[:10]
            
        except Exception as e:
            logger.error(f"Quantum jump prediction error: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Отримати поточну ціну"""
        try:
            url = f"{self.base_url}/ticker/price?symbol={symbol}"
            data = safe_request(url)
            if data and isinstance(data, dict) and 'price' in data:
                return float(data['price'])
            return 0
        except:
            return 0
    
    def calculate_volatility(self, symbol: str) -> float:
        """Розрахунок волатильності"""
        try:
            klines = safe_request(
                f"{self.base_url}/klines",
                params={'symbol': symbol, 'interval': '1h', 'limit': 24}
            )
            
            if klines and isinstance(klines, list):
                closes = [float(k[4]) for k in klines]
                if len(closes) >= 2:
                    returns = np.diff(closes) / closes[:-1]
                    return np.std(returns) * 100
            return 15.0  # Дефолтна волатильність
        except:
            return 15.0
    
    def calculate_quantum_target(self, current_price: float, direction: str, volatility: float) -> float:
        """Розрахунок квантової цільової ціни"""
        # Квантове тунелювання через рівні
        move_pct = volatility * 0.8 
        
            def calculate_quantum_target(self, current_price: float, direction: str, volatility: float) -> float:
        """Розрахунок квантової цільової ціни"""
        # Квантове тунелювання через рівні
        move_pct = volatility * 0.8  # 80% від волатильності
        
        if direction == 'UP':
            return current_price * (1 + move_pct / 100)
        else:
            return current_price * (1 - move_pct / 100)
    
    def quantum_timeframe(self, schrodinger_state: Dict) -> str:
        """Визначення квантового таймфрейму"""
        entropy = schrodinger_state.get('quantum_entropy', 0.5)
        
        if entropy > 0.7:
            return "15-30 хвилин"  # Висока ентропія - швидкі зміни
        elif entropy > 0.4:
            return "30-60 хвилин"  # Середня ентропія
        else:
            return "1-2 години"    # Низька ентропія - повільні зміни
    
    def calculate_quantum_risk(self, schrodinger_state: Dict) -> str:
        """Розрахунок квантового ризику"""
        superposition = schrodinger_state.get('superposition', 0.5)
        
        if superposition > 0.7:
            return "ВИСОКИЙ ⚠️"  # Сильна суперпозиція - невизначеність
        elif superposition > 0.4:
            return "СЕРЕДНІЙ 🟡"
        else:
            return "НИЗЬКИЙ 🟢"
    
    def generate_quantum_strategy(self, prediction: Dict) -> str:
        """Генерація квантової стратегії"""
        symbol = prediction['symbol']
        direction = prediction['direction']
        
        strategies = {
            'UP': [
                f"🟢 Квантова купівля {symbol}",
                f"⚡ Вхід: поточна ціна +0.3-0.7%",
                f"🎯 Ціль: {prediction['target_price']:.6f}",
                f"⏰ Час: {prediction['timeframe']}",
                f"📊 Ризик: {prediction['risk_level']}"
            ],
            'DOWN': [
                f"🔴 Квантовий шорт {symbol}",
                f"⚡ Вхід: поточна ціна -0.3-0.7%", 
                f"🎯 Ціль: {prediction['target_price']:.6f}",
                f"⏰ Час: {prediction['timeframe']}",
                f"📊 Ризик: {prediction['risk_level']}"
            ]
        }
        
        return "\n".join(strategies[direction])

# Глобальний екземпляр квантового предиктора
quantum_predictor = QuantumPredictor()