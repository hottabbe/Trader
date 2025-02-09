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
        self.lstm_model = None
        self.backtester = Backtester(self.model, self.risk_manager)
        self.all_data = pd.DataFrame()
        
        # Глобальный список required_columns
        self.required_columns = [
            'momentum', 'volatility', 'rsi_14', 'ema_20', 'macd', 'macd_signal', 
            'volume_profile', 'atr_14', 'upper_band', 'lower_band', 'adx_14', 'obv'
        ]
        
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
                    if self.all_data.empty:
                        data = self.data_fetcher.fetch_all_historical_data(symbol, self.timeframe)
                    else:
                        # Получаем данные, начиная с самой ранней временной метки
                        earliest_timestamp = self.all_data['timestamp'].min()
                        data = self.data_fetcher.fetch_all_historical_data(symbol, self.timeframe, since=int(earliest_timestamp.timestamp() * 1000))
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

                # Разделяем данные на обучающую и тестовую выборки
                train_data = self.all_data.iloc[:-100]  # Все данные, кроме последних 100 строк
                test_data = self.all_data.iloc[-100:]  # Последние 100 строк

                if train_data.empty or test_data.empty:
                    raise ValueError("Not enough data to create train and test sets.")

                log(f"Training data points: {len(train_data)}")
                log(f"Test data points: {len(test_data)}")

                # Обучаем модель на исторических данных
                X_train = train_data[self.required_columns].values
                y_train_direction = train_data["target_direction"].values
                y_train_level = train_data["target_level"].values

                log("Training the model")
                self.model.train(X_train, y_train_direction, y_train_level)

                # Оцениваем модель на тестовых данных
                X_test = test_data[self.required_columns].values
                y_test_direction = test_data["target_direction"].values
                y_test_level = test_data["target_level"].values

                accuracy, mse = self.model.evaluate(X_test, y_test_direction, y_test_level)
                log(f"Model evaluation completed with accuracy: {accuracy:.2%}, MSE: {mse:.4f}")

                # Сохраняем модель и данные в файл
                save_model(self.model, "brain.model", self.all_data, self.lstm_model)

        except Exception as e:
            log(f"Error during bot initialization: {e}", level="error")
            traceback.print_exc()
            raise

    def run(self):
        while True:
            try:
                for symbol in self.symbols:
                    log(f"Fetching latest data for {symbol}")
                    latest_data = self.data_fetcher.fetch_ohlcv(symbol, self.timeframe, limit=500)
                    if latest_data.empty:
                        log(f"No new data for {symbol}. Skipping.", level="warning")
                        continue

                    log(f"Creating features for {symbol}")
                    latest_data = self.feature_engineer.create_features(latest_data)
                    latest_data["symbol"] = symbol

                    # Проверяем, что все необходимые колонки присутствуют
                    if not all(col in latest_data.columns for col in self.required_columns):
                        log(f"Missing required columns for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[self.required_columns].isnull().any().any():
                        log(f"NaN values found in data for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[self.required_columns].empty:
                        log(f"No data available for prediction for {symbol}. Skipping.", level="warning")
                        continue

                    # Проводим бэктестинг перед принятием решения
                    if not self.all_data.empty:
                        accuracy, profit = self.backtester.run(self.all_data, self.deposit, self.risk_per_trade)
                        log(f"Backtesting accuracy for {symbol}: {accuracy:.2%}, Profit: {profit:.2f}%")

                        # Анализируем результаты бэктестинга
                        if accuracy < 0.5 or profit < 0:
                            log("Model performance is poor. Calibrating model...", level="warning")
                            self.calibrate_model()

                    # Проверяем, обучена ли модель
                    if not self.model.is_trained:
                        log("Model is not trained. Skipping prediction.", level="warning")
                        continue

                    # Получаем сигнал от модели
                    X = latest_data[self.required_columns].values
                    direction, level = self.model.predict(X)
                    signal = 1 if direction[0] == 1 else -1  # Преобразуем направление в сигнал (1 — лонг, -1 — шорт)

                    # Рассчитываем точки входа, стоп-лосса и тейк-профита
                    entry_price = latest_data["close"].iloc[-1]
                    atr = latest_data["atr_14"].iloc[-1]
                    stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, atr)

                    # Если сигнал на продажу, меняем местами стоп-лосс и тейк-профит
                    if signal == -1:
                        stop_loss, take_profit = take_profit, stop_loss

                    position_size = self.risk_manager.calculate_position_size(self.deposit, self.risk_per_trade, entry_price, stop_loss)

                    explanation = self.generate_explanation(latest_data, signal, entry_price, stop_loss, take_profit, position_size)
                    log(explanation)

                    # Добавляем новые данные и доучиваем модель
                    self.all_data = pd.concat([self.all_data, latest_data])
                    X = self.all_data[self.required_columns].values
                    y_direction = self.all_data["target_direction"].values
                    y_level = self.all_data["target_level"].values
                    self.model.train(X, y_direction, y_level)

                    # Перезаписываем модель
                    log("Updating model in brain.model")
                    save_model(self.model, "brain.model")

                time.sleep(60 * 60)

            except Exception as e:
                log(f"Error during bot execution: {e}", level="error")
                traceback.print_exc()
                time.sleep(60)

    def calibrate_model(self):
        """
        Калибрует модель на основе результатов бэктестинга.
        """
        try:
            log("Calibrating model...")

            # Добавляем больше данных для обучения
            for symbol in self.symbols:
                log(f"Fetching additional historical data for {symbol}")
                additional_data = self.data_fetcher.fetch_all_historical_data(symbol, self.timeframe)
                if additional_data.empty:
                    log(f"No additional data for {symbol}. Skipping.", level="warning")
                    continue

                # Преобразуем временные метки
                additional_data = convert_timestamps(additional_data)

                # Проверяем данные на корректность
                if not validate_data(additional_data):
                    log(f"Invalid data for {symbol}. Skipping.", level="warning")
                    continue

                log(f"Creating features for {symbol}")
                additional_data = self.feature_engineer.create_features(additional_data)
                if additional_data.empty:
                    log(f"No data after feature creation for {symbol}. Skipping.", level="warning")
                    continue

                additional_data["symbol"] = symbol
                self.all_data = pd.concat([self.all_data, additional_data])

            if self.all_data.empty:
                raise ValueError("No data available for model calibration.")

            # Логируем временной промежуток данных
            log_data_range(self.all_data)

            # Создаем целевую переменную
            self.all_data["target_direction"] = (self.all_data["close"].shift(-5) > self.all_data["close"]).astype(int)
            self.all_data["target_level"] = (self.all_data["close"].shift(-5) - self.all_data["close"]) / self.all_data["close"]

            # Разделяем данные на обучающую и тестовую выборки
            train_data = self.all_data.iloc[:-100]  # Все данные, кроме последних 100 строк
            test_data = self.all_data.iloc[-100:]  # Последние 100 строк

            if train_data.empty or test_data.empty:
                raise ValueError("Not enough data to create train and test sets.")

            log(f"Training data points: {len(train_data)}")
            log(f"Test data points: {len(test_data)}")

            # Обучаем модель на исторических данных
            X_train = train_data[self.required_columns].values
            y_train_direction = train_data["target_direction"].values
            y_train_level = train_data["target_level"].values

            log("Training the model")
            self.model.train(X_train, y_train_direction, y_train_level)

            # Оцениваем модель на тестовых данных
            X_test = test_data[self.required_columns].values
            y_test_direction = test_data["target_direction"].values
            y_test_level = test_data["target_level"].values

            accuracy, mse = self.model.evaluate(X_test, y_test_direction, y_test_level)
            log(f"Model accuracy on test data after calibration: {accuracy:.2%}, MSE: {mse:.4f}")

            # Сохраняем модель и данные в файл
            save_model(self.model, "brain.model", self.all_data, self.lstm_model)

        except Exception as e:
            log(f"Error during model calibration: {e}", level="error")
            traceback.print_exc()
                
    def generate_explanation(self, latest_data, signal, entry_price, stop_loss, take_profit, position_size):
        """
        Генерирует объяснение для торгового решения.
        """
        explanation = "Decision based on the following indicators:\n"
        explanation += f"- Momentum: {latest_data['momentum'].iloc[0]:.4f} (trend)\n"
        explanation += f"- Volatility: {latest_data['volatility'].iloc[0]:.4f} (risk)\n"
        explanation += f"- RSI 14: {latest_data['rsi_14'].iloc[0]:.2f} (overbought/oversold)\n"
        explanation += f"- EMA 20: {latest_data['ema_20'].iloc[0]:.2f} (mid-term trend)\n"
        explanation += f"- MACD: {latest_data['macd'].iloc[0]:.4f}, Signal: {latest_data['macd_signal'].iloc[0]:.4f} (trend dynamics)\n"
        explanation += f"- Volume Profile: {latest_data['volume_profile'].iloc[0]:.2f} (trading activity)\n"
        explanation += f"- ATR: {latest_data['atr'].iloc[0]:.2f} (volatility)\n"
        explanation += f"- Bollinger Bands: Upper {latest_data['upper_band'].iloc[0]:.2f}, Lower {latest_data['lower_band'].iloc[0]:.2f} (volatility boundaries)\n"
        explanation += f"- ADX: {latest_data['adx'].iloc[0]:.2f} (trend strength)\n"
        explanation += f"- OBV: {latest_data['obv'].iloc[0]:.2f} (volume flow)\n"
        explanation += "Signal: BUY (price expected to rise)\n" if signal == 1 else "Signal: SELL (price expected to fall)\n"
        explanation += f"Entry price: {entry_price:.2f}\n"
        explanation += f"TP: {take_profit:.2f} - Potential profit: {(take_profit - entry_price) * position_size:.2f}$\n"
        explanation += f"SL: {stop_loss:.2f} - Potential loss: {(entry_price - stop_loss) * position_size:.2f}$\n"
        return explanation