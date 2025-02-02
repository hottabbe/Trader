import pandas as pd
import os
import time
from utils import log, save_model, load_model, setup_logging, validate_data, convert_timestamps, log_data_range, prepare_lstm_data
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from risk_manager import RiskManager
from trading_model import TradingModel
from backtester import Backtester
from lstm_model import LSTMModel
import traceback

class TradingBot:
    def __init__(self, symbols, timeframe, deposit, risk_per_trade, news_api_key):
        self.symbols = symbols
        self.timeframe = timeframe
        self.deposit = deposit
        self.risk_per_trade = risk_per_trade
        self.data_fetcher = DataFetcher(news_api_key)
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        self.model = TradingModel()
        self.lstm_model = None  # Модель LSTM
        self.backtester = Backtester(self.model, self.risk_manager)
        self.all_data = pd.DataFrame()
        setup_logging()

    def initialize(self):
        try:
            log("Initializing bot")

            # Загружаем модель и данные, если файл существует
            if os.path.exists("brain.model"):
                log("Loading model and data from brain.model")
                self.model, self.all_data, self.lstm_model = load_model("brain.model")

                # Проверяем, что модель загружена и обучена
                if not self.model.is_trained:
                    log("Loaded model is not trained. Deleting brain.model and training a new model.", level="warning")
                    os.remove("brain.model")
                    self.model = TradingModel()  # Создаем новую модель
                    self.all_data = pd.DataFrame()  # Сбрасываем данные
                    self.lstm_model = None  # Сбрасываем LSTM
                else:
                    log("Model and data loaded successfully.")
            else:
                log("No existing model found. Training a new model.")
                self.all_data = pd.DataFrame()  # Инициализируем пустой DataFrame
                self.lstm_model = None  # Инициализируем LSTM

            # Если модель не загружена, обучаем с нуля
            if not self.model.is_trained:
                for symbol in self.symbols:
                    log(f"Fetching historical data for {symbol}")
                    data = self.data_fetcher.fetch_all_historical_data(symbol, self.timeframe)
                    if data.empty:
                        log(f"No data for {symbol}. Skipping.", level="warning")
                        continue

                    # Преобразуем временные метки
                    data = convert_timestamps(data)

                    # Проверяем данные на корректность
                    if not validate_data(data):
                        log(f"Invalid data for {symbol}. Skipping.", level="warning")
                        continue

                    log(f"Creating features for {symbol}")
                    data = self.feature_engineer.create_features(data)
                    if data.empty:
                        log(f"No data after feature creation for {symbol}. Skipping.", level="warning")
                        continue

                    data["symbol"] = symbol
                    self.all_data = pd.concat([self.all_data, data])

                if self.all_data.empty:
                    raise ValueError("No data available for model initialization.")

                # Логируем временной промежуток данных
                log_data_range(self.all_data)

                # Создаем целевую переменную
                self.all_data["signal"] = (self.all_data["return"].shift(-1) > 0).astype(int)

                # Разделяем данные на обучающую и тестовую выборки
                train_data = self.all_data.iloc[:-100]  # Все данные, кроме последних 100 строк
                test_data = self.all_data.iloc[-100:]  # Последние 100 строк

                if train_data.empty or test_data.empty:
                    raise ValueError("Not enough data to create train and test sets.")

                log(f"Training data points: {len(train_data)}")
                log(f"Test data points: {len(test_data)}")

                # Обучаем модель на исторических данных
                required_columns = ['momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 'volume_profile', 'atr', 'upper_band', 'lower_band', 'adx', 'obv']
                X_train = train_data[required_columns].values
                y_train = train_data["signal"].values

                log("Training the model")
                self.model.train(X_train, y_train)

                # Оцениваем модель на тестовых данных
                X_test = test_data[required_columns].values
                y_test = test_data["signal"].values

                accuracy = self.model.evaluate(X_test, y_test)
                log(f"Model accuracy on test data: {accuracy:.2%}")

                # Обучаем LSTM
                log("Preparing data for LSTM")
                X_lstm, y_lstm, scaler = prepare_lstm_data(self.all_data)
                self.lstm_model = LSTMModel(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
                self.lstm_model.train(X_lstm, y_lstm)

                # Сохраняем модель и данные в файл
                save_model(self.model, "brain.model", self.all_data, self.lstm_model)

        except Exception as e:
            log(f"Error during bot initialization: {e}", level="error")
            traceback.print_exc()  # Выводим полный стектрейс ошибки
            raise
        
    def run(self):
        while True:
            try:
                for symbol in self.symbols:
                    log(f"Fetching latest data for {symbol}")
                    latest_data = self.data_fetcher.fetch_ohlcv(symbol, self.timeframe, limit=500)  # Увеличили до 500 свечей
                    if latest_data.empty:
                        log(f"No new data for {symbol}. Skipping.", level="warning")
                        continue

                    log(f"Creating features for {symbol}")
                    latest_data = self.feature_engineer.create_features(latest_data)
                    latest_data["symbol"] = symbol

                    required_columns = [
                        'momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 
                        'volume_profile', 'atr', 'upper_band', 'lower_band', 'adx', 'obv'
                    ]
                    if not all(col in latest_data.columns for col in required_columns):
                        log(f"Missing required columns for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[required_columns].isnull().any().any():
                        log(f"NaN values found in data for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[required_columns].empty:
                        log(f"No data available for prediction for {symbol}. Skipping.", level="warning")
                        continue

                    # Проводим бэктестинг перед принятием решения
                    if not self.all_data.empty:
                        accuracy, profit = self.backtester.run(self.all_data)  # Получаем кортеж
                        log(f"Backtesting accuracy for {symbol}: {accuracy:.2%}, Profit: {profit:.2f}%")  # Форматируем оба значения

                    # Проверяем, обучена ли модель
                    if not self.model.is_trained:
                        log("Model is not trained. Skipping prediction.", level="warning")
                        continue

                    # Получаем сигнал от модели
                    X = latest_data[required_columns].values
                    signal = self.model.predict(X)[0]
                    signal = -1 if signal == 0 else 1

                    # Рассчитываем точки входа, стоп-лосса и тейк-профита
                    entry_price = latest_data["close"].iloc[-1]
                    stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, latest_data["atr"].iloc[-1])

                    # Если сигнал на продажу, меняем местами стоп-лосс и тейк-профит
                    if signal == -1:
                        stop_loss, take_profit = take_profit, stop_loss

                    position_size = self.risk_manager.calculate_position_size(self.deposit, self.risk_per_trade, entry_price, stop_loss)

                    explanation = self.generate_explanation(latest_data, signal, entry_price, stop_loss, take_profit, position_size)
                    log(explanation)

                    # Добавляем новые данные и доучиваем модель
                    self.all_data = pd.concat([self.all_data, latest_data])
                    X = self.all_data[required_columns].values
                    y = self.all_data["signal"].values
                    self.model.train(X, y)

                    # Перезаписываем модель
                    log("Updating model in brain.model")
                    save_model(self.model, "brain.model")

                time.sleep(60 * 60)

            except Exception as e:
                log(f"Error during bot execution: {e}", level="error")
                traceback.print_exc()
                time.sleep(60)