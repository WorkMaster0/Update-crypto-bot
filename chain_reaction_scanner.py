# chain_reaction_scanner.py
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils import safe_request
import numpy as np

logger = logging.getLogger(__name__)

class ChainReactionScanner:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.correlation_cache = {}
        self.last_analysis_time = None
        
    def analyze_correlation_network(self) -> Dict[str, float]:
        """Аналіз кореляційної мережі між токенами"""
        try:
            # Отримуємо дані за останні 24 години
            url = f"{self.base_url}/ticker/24hr"
            data = safe_request(url)
            
            if not data or not isinstance(data, list):
                return {}
            
            # Фільтруємо USDT пари з достатнім обсягом
            usdt_pairs = [
                d for d in data 
                if isinstance(d, dict) and 
                d.get('symbol', '').endswith('USDT') and 
                float(d.get('quoteVolume', 0)) > 10000000  # 10M+ обсяг
            ]
            
            # Беремо топ-30 за обсягом
            top_symbols = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:30]
            symbols = [s['symbol'] for s in top_symbols]
            
            # Отримуємо історичні дані для кореляційного аналізу
            correlation_matrix = self.calculate_correlation_matrix(symbols)
            
            # Знаходимо найсильніші кореляції
            strong_correlations = self.find_strong_correlations(correlation_matrix, symbols)
            
            return strong_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlation network: {e}")
            return {}
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Розрахунок матриці кореляцій між токенами"""
        try:
            # Отримуємо історичні дані (закриття за останні 7 днів)
            closes_matrix = []
            
            for symbol in symbols:
                klines = safe_request(
                    f"{self.base_url}/klines",
                    params={'symbol': symbol, 'interval': '1h', 'limit': 168}
                )
                
                if klines and isinstance(klines, list) and len(klines) >= 24:
                    closes = [float(k[4]) for k in klines[-24:]]  # Останні 24 години
                    closes_matrix.append(closes)
                else:
                    # Додаємо нулі якщо дані відсутні
                    closes_matrix.append([0] * 24)
            
            # Конвертуємо в numpy array для кореляцій
            matrix = np.array(closes_matrix)
            
            # Розраховуємо кореляційну матрицю
            correlation_matrix = np.corrcoef(matrix)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return np.zeros((len(symbols), len(symbols)))
    
    def find_strong_correlations(self, correlation_matrix: np.ndarray, symbols: List[str]) -> Dict[str, float]:
        """Знаходження сильних кореляцій між токенами"""
        strong_correlations = {}
        
        try:
            n = len(symbols)
            for i in range(n):
                for j in range(i + 1, n):
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) > 0.7:  # Сильна кореляція
                        pair_key = f"{symbols[i]}_{symbols[j]}"
                        strong_correlations[pair_key] = correlation
            
            return strong_correlations
            
        except Exception as e:
            logger.error(f"Error finding strong correlations: {e}")
            return {}
    
    def detect_chain_reactions(self) -> List[Dict]:
        """Детектування ланцюгових реакцій"""
        try:
            # Аналізуємо кореляційну мережу
            correlations = self.analyze_correlation_network()
            
            # Отримуємо поточні цінові зміни
            url = f"{self.base_url}/ticker/24hr"
            data = safe_request(url)
            
            if not data or not isinstance(data, list):
                return []
            
            # Знаходимо токени з найбільшими змінами
            price_changes = {}
            for item in data:
                if isinstance(item, dict) and item.get('symbol', '').endswith('USDT'):
                    symbol = item['symbol']
                    price_change = float(item.get('priceChangePercent', 0))
                    price_changes[symbol] = price_change
            
            # Шукаємо ланцюгові реакції
            chain_reactions = []
            
            # Аналізуємо кожну сильну кореляцію
            for corr_key, correlation in correlations.items():
                symbol1, symbol2 = corr_key.split('_')
                
                if symbol1 in price_changes and symbol2 in price_changes:
                    change1 = price_changes[symbol1]
                    change2 = price_changes[symbol2]
                    
                    # Перевіряємо чи є ланцюгова реакція
                    if abs(change1) > 5 and abs(change2) > 3 and np.sign(change1) == np.sign(change2):
                        # Знаходимо лідера (більша зміна)
                        leader = symbol1 if abs(change1) > abs(change2) else symbol2
                        follower = symbol2 if leader == symbol1 else symbol1
                        
                        chain_reactions.append({
                            'leader': leader,
                            'follower': follower,
                            'correlation': correlation,
                            'leader_change': change1 if leader == symbol1 else change2,
                            'follower_change': change2 if leader == symbol1 else change1,
                            'time_delay': self.estimate_time_delay(leader, follower),
                            'confidence': min(90, abs(correlation) * 100 * 0.8)
                        })
            
            # Сортуємо за впевненістю
            chain_reactions.sort(key=lambda x: x['confidence'], reverse=True)
            return chain_reactions[:10]  # Топ-10 реакцій
            
        except Exception as e:
            logger.error(f"Error detecting chain reactions: {e}")
            return []
    
    def estimate_time_delay(self, leader: str, follower: str) -> str:
        """Оцінка часової затримки між реакціями"""
        try:
            # Спрощена логіка оцінки затримки
            delays = {
                'BTCUSDT': '15-30 хв',
                'ETHUSDT': '10-20 хв', 
                'SOLUSDT': '5-15 хв',
                'AVAXUSDT': '5-12 хв',
                'default': '10-25 хв'
            }
            
            return delays.get(leader, delays.get(follower, delays['default']))
            
        except Exception as e:
            logger.error(f"Error estimating time delay: {e}")
            return '15-30 хв'
    
    def predict_next_movers(self, current_reactions: List[Dict]) -> List[Dict]:
        """Прогнозування наступних токенів для руху"""
        next_movers = []
        
        try:
            # Аналізуємо історичні патерни
            for reaction in current_reactions:
                leader = reaction['leader']
                
                # Знаходимо токени з сильною кореляцією до лідера
                correlated_symbols = self.find_correlated_to_leader(leader)
                
                for symbol, correlation in correlated_symbols.items():
                    if symbol != reaction['follower']:  # Виключаємо вже активні
                        next_movers.append({
                            'symbol': symbol,
                            'correlated_to': leader,
                            'correlation_strength': correlation,
                            'expected_delay': self.estimate_time_delay(leader, symbol),
                            'confidence': min(85, correlation * 100 * 0.7)
                        })
            
            # Сортуємо за впевненістю
            next_movers.sort(key=lambda x: x['confidence'], reverse=True)
            return next_movers[:5]  # Топ-5 прогнозів
            
        except Exception as e:
            logger.error(f"Error predicting next movers: {e}")
            return []
    
    def find_correlated_to_leader(self, leader: str) -> Dict[str, float]:
        """Знаходження токенів з сильною кореляцією до лідера"""
        # Спрощена версія - в реальності потрібен повний кореляційний аналіз
        correlations = {
            'BTCUSDT': {'ETHUSDT': 0.85, 'SOLUSDT': 0.78, 'AVAXUSDT': 0.72},
            'ETHUSDT': {'BTCUSDT': 0.85, 'SOLUSDT': 0.81, 'MATICUSDT': 0.68},
            'SOLUSDT': {'BTCUSDT': 0.78, 'ETHUSDT': 0.81, 'AVAXUSDT': 0.88},
            'AVAXUSDT': {'SOLUSDT': 0.88, 'BTCUSDT': 0.72, 'ATOMUSDT': 0.65}
        }
        
        return correlations.get(leader, {})

# Глобальний екземпляр сканера
chain_reaction_scanner = ChainReactionScanner()