import logging
from telebot import TeleBot
from telebot.types import Message
from utils.api_client import BinanceClient
from analyzers.pump_detector import detect_pump_activity, analyze_volume
from analyzers.volume_analyzer import detect_volume_anomaly

logger = logging.getLogger(__name__)
binance_client = BinanceClient()

def setup_analysis_commands(bot: TeleBot, settings):
    @bot.message_handler(commands=['pump_scan'])
    def pump_scan_handler(message: Message):
        """Сканування на памп активність"""
        try:
            msg = bot.send_message(message.chat.id, "🔍 Сканую на памп активність...")
            
            # Отримуємо дані з Binance
            ticker_data = binance_client.get_ticker_24hr()
            if not ticker_data:
                bot.edit_message_text("❌ Помилка отримання даних з Binance", message.chat.id, msg.message_id)
                return
            
            # Фільтруємо USDT пари з високим обсягом
            symbols = [
                d for d in ticker_data
                if isinstance(d, dict) and 
                d.get("symbol", "").endswith("USDT") and 
                float(d.get("quoteVolume", 0)) > settings['min_volume']
            ]
            
            # Сортуємо за % зміни ціни
            symbols = sorted(
                symbols,
                key=lambda x: abs(float(x.get("priceChangePercent", 0))),
                reverse=True
            )
            
            top_symbols = [s["symbol"] for s in symbols[:settings['top_symbols']]]
            pump_signals = []
            
            for symbol in top_symbols:
                try:
                    df = binance_client.get_klines(symbol, interval="1h", limit=200)
                    if not df or len(df.get("c", [])) < 50:
                        continue
                    
                    closes = [float(c) for c in df["c"]]
                    volumes = [float(v) for v in df["v"]]
                    
                    # Детектуємо памп активність
                    pump_type, price_change, pump_data = detect_pump_activity(
                        symbol, closes, volumes, settings
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
    def volume_anomaly_handler(message: Message):
        """Пошук аномальних обсягів торгів"""
        try:
            msg = bot.send_message(message.chat.id, "🔍 Шукаю аномальні обсяги...")
            
            ticker_data = binance_client.get_ticker_24hr()
            if not ticker_data:
                bot.edit_message_text("❌ Помилка отримання даних", message.chat.id, msg.message_id)
                return
            
            # Фільтруємо та сортуємо символи
            symbols = [
                d for d in ticker_data
                if isinstance(d, dict) and 
                d.get("symbol", "").endswith("USDT") and 
                float(d.get("quoteVolume", 0)) > settings['min_volume'] / 10  # Нижчий поріг для аномалій
            ]
            
            symbols = sorted(
                symbols,
                key=lambda x: float(x.get("quoteVolume", 0)),
                reverse=True
            )
            
            top_symbols = [s["symbol"] for s in symbols[:50]]  # Більше символів для аналізу
            anomalies = []
            
            for symbol in top_symbols:
                try:
                    df = binance_client.get_klines(symbol, interval="1h", limit=100)
                    if not df or len(df.get("v", [])) < 24:
                        continue
                    
                    volumes = [float(v) for v in df["v"]]
                    
                    # Шукаємо аномалії обсягу
                    is_anomaly, anomaly_data = detect_volume_anomaly(symbol, volumes, settings)
                    
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
    def advanced_analysis_handler(message: Message):
        """Розширений аналіз обраного токена"""
        try:
            # Перевіряємо, чи вказано токен
            parts = message.text.split()
            if len(parts) < 2:
                bot.reply_to(message, "ℹ️ Використання: /advanced_analysis BTC")
                return
                
            symbol = parts[1].upper() + "USDT"
            msg = bot.send_message(message.chat.id, f"🔍 Аналізую {symbol}...")
            
            # Отримуємо дані
            df = binance_client.get_klines(symbol, interval="1h", limit=200)
            if not df or len(df.get("c", [])) < 50:
                bot.edit_message_text("❌ Не вдалося отримати дані для цього токена", message.chat.id, msg.message_id)
                return
            
            closes = [float(c) for c in df["c"]]
            volumes = [float(v) for v in df["v"]]
            last_price = closes[-1]
            
            # Виконуємо різні види аналізу
            pump_type, price_change, pump_data = detect_pump_activity(symbol, closes, volumes, settings)
            is_volume_anomaly, volume_data = detect_volume_anomaly(symbol, volumes, settings)
            volume_metrics = analyze_volume(volumes, settings)
            
            # Формуємо звіт
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
            
            # Додаємо рекомендацію
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