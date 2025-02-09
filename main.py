from trading_bot import TradingBot
import threading
import time

# Настройки
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
TIMEFRAME = '1h'
DEPOSIT = 1000
RISK_PER_TRADE = 0.01

# Инициализация бота
bot = TradingBot(SYMBOLS, TIMEFRAME, DEPOSIT, RISK_PER_TRADE)
bot.initialize()

def run_bot():
    """Запускает бота в бесконечном цикле с паузой между итерациями."""
    while True:
        bot.run()
        time.sleep(60 * 60)  # Пауза между итерациями (например, 1 час)

# Запуск бота в отдельном потоке
threading.Thread(target=run_bot).start()

# Основной поток может быть использован для других задач или просто ожидания
while True:
    time.sleep(1)