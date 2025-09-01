# app/analytics/levels.py
import numpy as np
from typing import Dict, List, Tuple
from app.config import PIVOT_LEFT_RIGHT, MAX_LEVELS

def _pivot_high(arr: np.ndarray, idx: int, left: int, right: int) -> bool:
    """Чи є arr[idx] локальним максимумом."""
    if idx - left < 0 or idx + right >= len(arr):
        return False
    return all(arr[idx] > arr[idx - i] for i in range(1, left + 1)) and \
           all(arr[idx] >= arr[idx + i] for i in range(1, right + 1))

def _pivot_low(arr: np.ndarray, idx: int, left: int, right: int) -> bool:
    """Чи є arr[idx] локальним мінімумом."""
    if idx - left < 0 or idx + right >= len(arr):
        return False
    return all(arr[idx] < arr[idx - i] for i in range(1, left + 1)) and \
           all(arr[idx] <= arr[idx + i] for i in range(1, right + 1))

def _cluster_levels(levels: List[float], max_levels: int = MAX_LEVELS, tolerance: float = 0.002) -> List[float]:
    """
    Кластеризує рівні (щоб не було 10 рівнів у +/- 0.1% зоні).
    tolerance = 0.002 → 0.2% відносний допуск.
    """
    if not levels:
        return []
    levels = sorted(set(levels))
    clustered = [levels[0]]

    for lvl in levels[1:]:
        if abs(lvl - clustered[-1]) / clustered[-1] > tolerance:
            clustered.append(lvl)

    # якщо забагато рівнів → вибираємо "найсильніші" (де ціна торкалась частіше)
    if len(clustered) > max_levels:
        # Тут можна зробити більш складний відбір, але залишимо топ по відстані від середнього
        mid = np.mean(clustered)
        clustered = sorted(clustered, key=lambda x: abs(x - mid))[:max_levels]

    return clustered

def find_levels(data: Dict[str, np.ndarray],
                left: int = PIVOT_LEFT_RIGHT,
                right: int = PIVOT_LEFT_RIGHT,
                max_levels: int = MAX_LEVELS) -> Dict[str, List[float]]:
    """
    Шукає рівні підтримки та опору на основі pivot-high/pivot-low.
    Використовує кластеризацію, щоб видалити надлишкові рівні.

    Args:
        data: {"t","o","h","l","c"}
        left/right: кількість сусідів для визначення pivot
        max_levels: обмеження кількості повернених рівнів

    Returns:
        {"supports": [...], "resistances": [...]}
    """
    highs, lows = data["h"], data["l"]
    resistances, supports = [], []

    for i in range(len(highs)):
        if _pivot_high(highs, i, left, right):
            resistances.append(highs[i])
        if _pivot_low(lows, i, left, right):
            supports.append(lows[i])

    return {
        "supports": _cluster_levels(supports, max_levels),
        "resistances": _cluster_levels(resistances, max_levels)
    }