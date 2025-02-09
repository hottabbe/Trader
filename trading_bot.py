import pandas as pd
import os
import time
from utils import log, save_model, load_model, setup_logging, validate_data, convert_timestamps, log_data_range, prepare_lstm_data,save_data,load_data
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from risk_manager import RiskManager
from trading_model import TradingModel
from backtester import Backtester
from lstm_model import LSTMModel
import traceback

class TradingBot:
    def __init__(self, symbols, timeframe, deposit, risk_per_trade):
        self.symbols = symbols
        self.timeframe = timeframe
        self.deposit = deposit
        self.risk_per_trade = risk_per_trade
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        self.models = {}  # Словарь для хранения моделей RandomForest для каждого символа
        self.lstm_models = {}  # Словарь для хранения LSTM-моделей для каждого символа
        self.required_columns = [
            'momentum', 'volatility', 'rsi_14', 'ema_20', 'macd', 'macd_signal', 
            'volume_profile', 'atr_14', 'upper_band', 'lower_band', 'adx_14', 'obv'
        ]
        self.backtester = Backtester(self.models,self.lstm_models,self.risk_manager,self.required_columns)
        self.all_data = {}  # Словарь для хранения данных по символам
        
        setup_logging()
        
    def initialize(self):
        try:
            log("Initializing bot")

            # Загружаем модели, если файлы существуют
            for symbol in self.symbols:
                model_filename = f"model_{symbol.replace('/', '_')}.pkl"
                lstm_model_filename = f"lstm_model_{symbol.replace('/', '_')}.pkl"

                if os.path.exists(model_filename):
                    log(f"Loading model for {symbol} from {model_filename}")
                    self.models[symbol], _, _ = load_model(model_filename)
                else:
                    log(f"No existing model found for {symbol}. Training a new model.")
                    self.models[symbol] = TradingModel()

                if os.path.exists(lstm_model_filename):
                    log(f"Loading LSTM model for {symbol} from {lstm_model_filename}")
                    self.lstm_models[symbol] = load_model(lstm_model_filename)
                else:
                    log(f"No existing LSTM model found for {symbol}. Training a new model.")
                    self.lstm_models[symbol] = LSTMModel(input_shape=(50, 5))  # Пример входной формы

            # Загружаем данные для каждого символа
            for symbol in self.symbols:
                data_filename = f"data_{symbol.replace('/', '_')}.pkl"
                if os.path.exists(data_filename):
                    log(f"Loading data for {symbol} from {data_filename}")
                    self.all_data[symbol] = load_data(data_filename)
                else:
                    log(f"No existing data found for {symbol}. Fetching historical data.")
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

                    # Сохраняем данные в файл
                    save_data(data, data_filename)
                    self.all_data[symbol] = data

            # Если данные для всех символов отсутствуют, выбрасываем ошибку
            if not self.all_data:
                raise ValueError("No data available for model initialization.")

            # Создаем индикаторы для каждого символа
            for symbol, data in self.all_data.items():
                log(f"Creating features for {symbol}")
                data = self.feature_engineer.create_features(data)
                if data.empty:
                    log(f"No data after feature creation for {symbol}. Skipping.", level="warning")
                    continue

                self.all_data[symbol] = data

            # Логируем временной промежуток данных
            for symbol, data in self.all_data.items():
                log_data_range(data)

            # Обучаем модели для каждого символа отдельно
            for symbol, data in self.all_data.items():
                log(f"Training models for {symbol}")

                # Разделяем данные на обучающую и тестовую выборки
                train_data = data.iloc[:-100]  # Все данные, кроме последних 100 строк
                test_data = data.iloc[-100:]  # Последние 100 строк

                if train_data.empty or test_data.empty:
                    log(f"Not enough data to create train and test sets for {symbol}. Skipping.", level="warning")
                    continue

                log(f"Training data points for {symbol}: {len(train_data)}")
                log(f"Test data points for {symbol}: {len(test_data)}")

                # Обучаем RandomForest
                X_train = train_data[self.required_columns].values
                y_train_direction = train_data["target_direction"].values
                y_train_level = train_data["target_level"].values

                log(f"Training RandomForest model for {symbol}")
                self.models[symbol].train(X_train, y_train_direction, y_train_level)

                # Обучаем LSTM
                X_train_lstm, y_train_lstm, _ = prepare_lstm_data(train_data)
                log(f"Training LSTM model for {symbol}")
                self.lstm_models[symbol].train(X_train_lstm, y_train_lstm)

                # Оцениваем модели на тестовых данных
                X_test = test_data[self.required_columns].values
                y_test_direction = test_data["target_direction"].values
                y_test_level = test_data["target_level"].values

                accuracy, mse = self.models[symbol].evaluate(X_test, y_test_direction, y_test_level)
                log(f"RandomForest evaluation for {symbol} completed with accuracy: {accuracy:.2%}, MSE: {mse:.4f}")

                X_test_lstm, y_test_lstm, _ = prepare_lstm_data(test_data)
                lstm_loss = self.lstm_models[symbol].model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
                log(f"LSTM evaluation for {symbol} completed with loss: {lstm_loss:.4f}")

                # Сохраняем модели в файлы
                model_filename = f"model_{symbol.replace('/', '_')}.pkl"
                lstm_model_filename = f"lstm_model_{symbol.replace('/', '_')}.pkl"
                save_model(self.models[symbol], model_filename, None, None)
                save_model(self.lstm_models[symbol], lstm_model_filename, None, None)

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
                    accuracy, profit = self.backtester.run(self.all_data, self.deposit, self.risk_per_trade)
                    log(f"Backtesting accuracy for {symbol}: {accuracy:.2%}, Profit: {profit:.2f}%")

                    # Анализируем результаты бэктестинга
                    if accuracy < 0.5 or profit < 0:
                        log("Model performance is poor. Calibrating model...", level="warning")
                        self.calibrate_model()

                    # Получаем сигнал от RandomForest
                    X = latest_data[self.required_columns].values
                    direction, level = self.models[symbol].predict(X)
                    signal_rf = 1 if direction[0] == 1 else -1  # Преобразуем направление в сигнал (1 — лонг, -1 — шорт)

                    # Получаем сигнал от LSTM
                    X_lstm, _, _ = prepare_lstm_data(latest_data)
                    predicted_price = self.lstm_models[symbol].predict(X_lstm)
                    signal_lstm = 1 if predicted_price > latest_data["close"].iloc[-1] else -1

                    # Объединяем сигналы (например, среднее значение)
                    signal = (signal_rf + signal_lstm) / 2

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
                    if symbol in self.all_data:
                        self.all_data[symbol] = pd.concat([self.all_data[symbol], latest_data])
                    else:
                        self.all_data[symbol] = latest_data

                    # Объединяем данные для обучения модели
                    combined_data = pd.concat(list(self.all_data.values()), ignore_index=True)

                    # Обучаем модель на объединенных данных
                    X = combined_data[self.required_columns].values
                    y_direction = combined_data["target_direction"].values
                    y_level = combined_data["target_level"].values
                    self.models[symbol].train(X, y_direction, y_level)

                    # Перезаписываем модель
                    log("Updating model in brain.model")
                    save_model(self.models[symbol], "brain.model")

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
                if symbol in self.all_data:
                    self.all_data[symbol] = pd.concat([self.all_data[symbol], additional_data])
                else:
                    self.all_data[symbol] = additional_data

            if not self.all_data:
                raise ValueError("No data available for model calibration.")

            # Логируем временной промежуток данных
            for symbol, data in self.all_data.items():
                log_data_range(data)

            # Объединяем данные для обучения модели
            combined_data = pd.concat(list(self.all_data.values()), ignore_index=True)

            # Создаем целевую переменную
            combined_data["target_direction"] = (combined_data["close"].shift(-5) > combined_data["close"]).astype(int)
            combined_data["target_level"] = (combined_data["close"].shift(-5) - combined_data["close"]) / combined_data["close"]

            # Разделяем данные на обучающую и тестовую выборки
            train_data = combined_data.iloc[:-100]  # Все данные, кроме последних 100 строк
            test_data = combined_data.iloc[-100:]  # Последние 100 строк

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

            # Сохраняем модель в файл
            save_model(self.model, "brain.model", None, self.lstm_model)

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
        explanation += f"- ADX: {latest_data['adx_14'].iloc[0]:.2f} (trend strength)\n"
        explanation += f"- OBV: {latest_data['obv'].iloc[0]:.2f} (volume flow)\n"
        explanation += "Signal: BUY (price expected to rise)\n" if signal == 1 else "Signal: SELL (price expected to fall)\n"
        explanation += f"Entry price: {entry_price:.2f}\n"
        explanation += f"TP: {take_profit:.2f} - Potential profit: {(take_profit - entry_price) * position_size:.2f}$\n"
        explanation += f"SL: {stop_loss:.2f} - Potential loss: {(entry_price - stop_loss) * position_size:.2f}$\n"
        return explanation