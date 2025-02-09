Торговый бот, использующий машинное обучение для принятия решений о покупке и продаже криптовалют на основе исторических данных. Бот использует модели RandomForest и LSTM для предсказания движения цены и управления рисками.

## Основные функции

- **Получение исторических данных**: Бот загружает исторические данные с биржи для каждой торговой пары.
- **Создание индикаторов**: На основе исторических данных создаются технические индикаторы, такие как RSI, MACD, ATR и другие.
- **Обучение моделей**: Для каждой торговой пары обучаются отдельные модели RandomForest и LSTM.
- **Бэктестинг**: Бот проводит бэктестинг на исторических данных, чтобы оценить точность моделей и потенциальную прибыль.
- **Торговые решения**: Бот принимает решения о покупке или продаже на основе предсказаний моделей и управления рисками.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-username/trading-bot.git
   cd trading-bot
   ```

2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Настройте параметры бота в файле `main.py`:
   ```python
   SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']  # Торговые пары
   TIMEFRAME = '1h'  # Таймфрейм (например, '1h', '1d')
   DEPOSIT = 1000  # Начальный депозит
   RISK_PER_TRADE = 0.01  # Риск на сделку (в процентах от депозита)
   ```

4. Запустите бота:
   ```bash
   python main.py
   ```

## Структура проекта

- **trading_bot.py**: Основной класс бота, управляющий всеми процессами.
- **data_fetcher.py**: Получение исторических данных с биржи.
- **feature_engineer.py**: Создание технических индикаторов на основе исторических данных.
- **trading_model.py**: Модель RandomForest для предсказания направления и уровня изменения цены.
- **lstm_model.py**: Модель LSTM для прогнозирования цен.
- **risk_manager.py**: Управление рисками, расчет стоп-лосса, тейк-профита и размера позиции.
- **backtester.py**: Бэктестинг моделей на исторических данных.
- **utils.py**: Вспомогательные функции для логирования, сохранения и загрузки моделей.
- **main.py**: Запуск бота.

## Используемые библиотеки

- **pandas**: Для работы с данными.
- **numpy**: Для численных вычислений.
- **scikit-learn**: Для обучения моделей RandomForest.
- **tensorflow**: Для обучения LSTM-моделей.
- **ccxt**: Для получения данных с биржи.
- **tulipy**: Для расчета технических индикаторов.

## Логирование

Бот использует логирование для отслеживания своей работы. Логи сохраняются в файл `trading_bot.log` и выводятся в консоль. Уровень логирования можно настроить в файле `utils.py`.

## Пример работы

1. Бот загружает исторические данные для каждой торговой пары.
2. Создаются технические индикаторы.
3. Обучаются модели RandomForest и LSTM.
4. Проводится бэктестинг для оценки точности моделей.
5. Бот принимает торговые решения на основе предсказаний моделей и управления рисками.

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

---

Если у вас есть вопросы или предложения, пожалуйста, создайте issue или свяжитесь со мной.
```

---
