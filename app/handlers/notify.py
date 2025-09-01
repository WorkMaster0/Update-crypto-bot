# handlers/notify.py
from telebot import types
from datetime import datetime
import re
from app.bot import bot
from app.analytics import send_test_notification
from handlers.ai_alert import generate_ai_signal

# ---------- Глобальні змінні ----------
notify_settings = {}       # Налаштування сповіщень користувачів
user_settings_state = {}   # Стан користувача для текстового вводу

# ---------- CALLBACK HANDLERS ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith("type_"))
def handle_signal_types(call):
    user_id = call.from_user.id
    type_map = {
        "type_all": ["ALL"],
        "type_breakout": ["BREAKOUT"],
        "type_trend": ["TREND"],
        "type_squeeze": ["SQUEEZE"]
    }
    if user_id not in notify_settings:
        notify_settings[user_id] = {"enabled": True}
    notify_settings[user_id]['signal_types'] = type_map.get(call.data, ["ALL"])
    bot.answer_callback_query(call.id, f"✅ Тип сигналів оновлено: {notify_settings[user_id]['signal_types']}")
    show_config_menu(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("remove_"))
def remove_favorite_callback(call):
    user_id = call.from_user.id
    symbol = call.data.replace("remove_", "")
    if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
        if symbol in notify_settings[user_id]['favorite_coins']:
            notify_settings[user_id]['favorite_coins'].remove(symbol)
            bot.answer_callback_query(call.id, f"✅ {symbol} видалено")
        else:
            bot.answer_callback_query(call.id, f"❌ {symbol} не знайдено")
    show_favorites_menu(call)

@bot.callback_query_handler(func=lambda call: call.data == "clear_all")
def clear_all_favorites(call):
    user_id = call.from_user.id
    if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
        notify_settings[user_id]['favorite_coins'] = []
        bot.answer_callback_query(call.id, "✅ Список очищено")
    show_favorites_menu(call)

# ---------- SHOW MENUS ----------
def show_config_menu(call):
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("🎯 Впевненість", callback_data="config_confidence"),
        types.InlineKeyboardButton("📊 Типи сигналів", callback_data="config_types")
    )
    markup.row(
        types.InlineKeyboardButton("⏰ Час активності", callback_data="config_time"),
        types.InlineKeyboardButton("💎 Улюблені монети", callback_data="config_favorites")
    )
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_back"))

    response = "⚙️ <b>Налаштування сповіщень:</b>\n\nОберіть опцію для зміни:"
    try:
        bot.edit_message_text(call.message.chat.id, call.message.message_id, response, parse_mode="HTML", reply_markup=markup)
    except:
        bot.send_message(call.message.chat.id, response, parse_mode="HTML", reply_markup=markup)

def show_signal_types_menu(call):
    user_id = call.from_user.id
    current_types = notify_settings.get(user_id, {}).get('signal_types', ['ALL'])
    response = ["📊 <b>Типи сигналів:</b>"]
    response += [f"{'✅' if t in current_types else '⚪️'} {t}" for t in ['ALL','BREAKOUT','TREND','SQUEEZE']]

    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("✅ ВСІ", callback_data="type_all"),
        types.InlineKeyboardButton("🚀 ПРОБОЇ", callback_data="type_breakout")
    )
    markup.row(
        types.InlineKeyboardButton("📈 ТРЕНДИ", callback_data="type_trend"),
        types.InlineKeyboardButton("🔍 СКВІЗИ", callback_data="type_squeeze")
    )
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config"))

    try:
        bot.edit_message_text(call.message.chat.id, call.message.message_id, "\n".join(response), parse_mode="HTML", reply_markup=markup)
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), parse_mode="HTML", reply_markup=markup)

def show_favorites_menu(call):
    user_id = call.from_user.id
    favorites = notify_settings.get(user_id, {}).get('favorite_coins', [])
    response = ["💎 <b>Улюблені монети:</b>\n"]
    markup = types.InlineKeyboardMarkup()
    if favorites:
        for coin in favorites:
            response.append(f"• {coin}")
            markup.add(types.InlineKeyboardButton(f"❌ Видалити {coin}", callback_data=f"remove_{coin}"))
        response.append("\n🎯 Натисніть на монету для видалення")
    else:
        response.append("• Список порожній")

    markup.row(types.InlineKeyboardButton("🗑️ Очистити всі", callback_data="clear_all"))
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config"))

    try:
        bot.edit_message_text(call.message.chat.id, call.message.message_id, "\n".join(response), parse_mode="HTML", reply_markup=markup)
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), parse_mode="HTML", reply_markup=markup)

# ---------- HANDLE TEXT INPUT ----------
@bot.message_handler(func=lambda m: True)
def handle_text_messages(message):
    user_id = message.from_user.id
    text = message.text.strip().lower()

    # Очистка улюблених через текст
    if text == "clear":
        if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
            notify_settings[user_id]['favorite_coins'] = []
            bot.send_message(user_id, "✅ Список улюблених очищено!")
        else:
            bot.send_message(user_id, "❌ Список вже порожній")
        return

    # Перевірка стану користувача
    if user_id in user_settings_state:
        state, callback_message = user_settings_state[user_id]
        if state == "waiting_confidence":
            try:
                val = int(text)
                if 50 <= val <= 90:
                    notify_settings.setdefault(user_id, {})['min_confidence'] = val
                    bot.send_message(user_id, f"✅ Мінімальна впевненість: {val}%")
                    show_config_menu(callback_message)
                else:
                    bot.send_message(user_id, "❌ Введіть число 50-90")
            except:
                bot.send_message(user_id, "❌ Введіть число")
        elif state == "waiting_favorites":
            coins = [c.strip().upper() for c in text.split(",") if c.strip().endswith("USDT")]
            if coins:
                notify_settings.setdefault(user_id, {})['favorite_coins'] = coins
                bot.send_message(user_id, f"✅ Улюблені монети: {', '.join(coins)}")
            else:
                bot.send_message(user_id, "❌ Невірний формат. Приклад: BTCUSDT,ETHUSDT")
            show_config_menu(callback_message)
            del user_settings_state[user_id]