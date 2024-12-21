import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Укажите ваш токен Telegram-бота, полученный от BotFather
API_TOKEN = 'ВАШ_ТОКЕН'

# Название модели из Hugging Face
MODEL_NAME = "distilgpt2"

# Загружаем модель и токенизатор
print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Модель загружена!")

# Создаем объект бота
bot = telebot.TeleBot(API_TOKEN)

# Функция генерации ответа
def generate_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я Devil — бот с бесплатной ИИ-моделью. Напиши что-нибудь, и я отвечу!")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def reply_to_user(message):
    user_input = message.text
    try:
        response = generate_response(user_input)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, "Произошла ошибка. Попробуй снова позже.")
        print(f"Ошибка: {e}")

# Запуск бота
print("Бот запущен. Нажмите Ctrl+C для остановки.")
bot.polling()
