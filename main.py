from trading_bot import TradingBot
import threading
import time

# Настройки
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
timeframe = '1h'
deposit = 1000
risk_per_trade = 0.01
news_api_key = '3426128c3e854e9798b80603dee3b101'

# Инициализация бота
bot = TradingBot(symbols, timeframe, deposit, risk_per_trade, news_api_key)
bot.initialize()

# Функция для запуска бота в бесконечном цикле
def run_bot():
    while True:
        bot.run()
        time.sleep(60 * 60)  # Пауза между итерациями (например, 1 час)

# Запуск бота в отдельном потоке
threading.Thread(target=run_bot).start()

# Основной поток может быть использован для других задач или просто ожидания
while True:
    time.sleep(1)