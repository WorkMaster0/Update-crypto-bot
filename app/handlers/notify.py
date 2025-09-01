# app/handlers/notify.py
from telebot import types
from app.bot import bot
from app.handlers.ai_alert import generate_ai_signal
from datetime import datetime
import re

# Глобальні змінні для збереження налаштувань
notify_settings = {}
user_settings_state = {}  # стан користувача для текстового вводу

# ---------- AI SIGNAL NOTIFICATION ----------
def send_test_notification(user_id: int):
    """Відправити тестове сповіщення з AI сигналом"""
    # Приклад з BTCUSDT
    signal = generate_ai_signal("BTCUSDT")
    text = [
        "🎯 <b>TEST AI ALERT</b>",
        signal["signal_text"],
        "",
        f"⏰ {datetime.utcnow().strftime('%H:%M:%S')} UTC",
        "",
        "💡 Ви будете отримувати сповіщення про найкращі торгові можливості!"
    ]
    bot.send_message(user_id, "\n".join(text), parse_mode="HTML")

# ---------- SETTINGS MENU ----------
def show_config_menu(call):
    """Показати меню налаштувань"""
    user_id = call.from_user.id
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("🎯 Впевненість", callback_data="config_confidence"),
        types.InlineKeyboardButton("📊 Типи сигналів", callback_data="config_types")
    )
    markup.row(
        types.InlineKeyboardButton("⏰ Час активності", callback_data="config_time"),
        types.InlineKeyboardButton("💎 Улюблені монети", callback_data="config_favorites")
    )
    markup.row(
        types.InlineKeyboardButton("🔙 Назад", callback_data="notify_back")
    )

    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="⚙️ <b>Налаштування сповіщень:</b>\n\nОберіть опцію для зміни:",
        parse_mode="HTML",
        reply_markup=markup
    )

# ---------- SIGNAL TYPES MENU ----------
def show_signal_types_menu(call):
    """Меню типів сигналів"""
    user_id = call.from_user.id
    current_types = notify_settings.get(user_id, {}).get("signal_types", ["ALL"])
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("✅ ВСІ", callback_data="type_all"),
        types.InlineKeyboardButton("🚀 ПРОБОЇ", callback_data="type_breakout")
    )
    markup.row(
        types.InlineKeyboardButton("📈 ТРЕНДИ", callback_data="type_trend"),
        types.InlineKeyboardButton("🔍 СКВІЗИ", callback_data="type_squeeze")
    )
    markup.row(
        types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config")
    )
    txt = ["📊 <b>Оберіть типи сигналів:</b>\n"]
    for t in ["ALL","BREAKOUT","TREND","SQUEEZE"]:
        txt.append("✅ "+t if t in current_types else "⚪️ "+t)
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="\n".join(txt),
        parse_mode="HTML",
        reply_markup=markup
    )

# ---------- FAVORITES MENU ----------
def show_favorites_menu(call):
    """Меню улюблених монет"""
    user_id = call.from_user.id
    favorites = notify_settings.get(user_id, {}).get("favorite_coins", [])
    markup = types.InlineKeyboardMarkup()
    response = ["💎 <b>Улюблені монети:</b>\n"]
    if favorites:
        for coin in favorites:
            response.append(f"• {coin}")
            markup.add(types.InlineKeyboardButton(f"❌ Видалити {coin}", callback_data=f"remove_{coin}"))
        markup.row(types.InlineKeyboardButton("🗑️ Очистити всі", callback_data="clear_all"))
    else:
        response.append("• Список порожній")
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config"))
    bot.edit_message_text(
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        text="\n".join(response),
        parse_mode="HTML",
        reply_markup=markup
    )

# ---------- CALLBACK HANDLERS ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith("type_"))
def handle_type_selection(call):
    user_id = call.from_user.id
    mapping = {
        "type_all": ["ALL"],
        "type_breakout": ["BREAKOUT"],
        "type_trend": ["TREND"],
        "type_squeeze": ["SQUEEZE"]
    }
    selected = mapping.get(call.data, ["ALL"])
    if user_id not in notify_settings:
        notify_settings[user_id] = {"enabled": True}
    notify_settings[user_id]["signal_types"] = selected
    bot.answer_callback_query(call.id, "✅ Типи сигналів оновлено")
    show_config_menu(call)

@bot.callback_query_handler(func=lambda call: call.data.startswith("remove_"))
def remove_favorite_callback(call):
    user_id = call.from_user.id
    coin = call.data.replace("remove_","")
    if user_id in notify_settings and "favorite_coins" in notify_settings[user_id]:
        if coin in notify_settings[user_id]["favorite_coins"]:
            notify_settings[user_id]["favorite_coins"].remove(coin)
            bot.answer_callback_query(call.id, f"✅ {coin} видалено")
    show_favorites_menu(call)

@bot.callback_query_handler(func=lambda call: call.data == "clear_all")
def clear_all_favorites(call):
    user_id = call.from_user.id
    if user_id in notify_settings:
        notify_settings[user_id]["favorite_coins"] = []
    bot.answer_callback_query(call.id, "✅ Список очищено")
    show_favorites_menu(call)