import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

# Налаштування аналізу
ANALYSIS_SETTINGS = {
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

# Налаштування сповіщень
ALERT_SETTINGS = {
    'pump_alert_enabled': True,
    'dump_alert_enabled': True,
    'volume_alert_enabled': True,
    'check_interval_minutes': 30
}