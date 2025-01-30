import pandas as pd
import time
import numpy as np
import os
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from risk_manager import RiskManager
from trading_model import TradingModel
from backtester import Backtester
from utils import log, setup_logging, save_model, load_model

class TradingBot:
    def __init__(self, symbols, timeframe, deposit, risk_per_trade, news_api_key, create_logs=True):
        self.symbols = symbols
        self.timeframe = timeframe
        self.deposit = deposit
        self.risk_per_trade = risk_per_trade
        self.data_fetcher = DataFetcher(news_api_key)
        self.feature_engineer = FeatureEngineer()
        self.risk_manager = RiskManager()
        self.model = TradingModel()
        self.backtester = Backtester(self.model)
        self.all_data = pd.DataFrame()
        setup_logging(create_logs)

    def initialize(self):
        try:
            log("Initializing bot")

            # Загружаем модель, если файл существует
            if os.path.exists("brain.model"):
                log("Loading model from brain.model")
                self.model = load_model("brain.model")
            else:
                log("No existing model found. Training a new model.")

                for symbol in self.symbols:
                    log(f"Fetching historical data for {symbol}")
                    data = self.data_fetcher.fetch_all_historical_data(symbol, self.timeframe)
                    if data.empty:
                        log(f"No data for {symbol}. Skipping.", level="warning")
                        continue

                    log(f"Creating features for {symbol}")
                    data = self.feature_engineer.create_features(data)
                    data["symbol"] = symbol
                    self.all_data = pd.concat([self.all_data, data])

                if self.all_data.empty:
                    raise ValueError("No data available for model initialization.")

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

                # Сохраняем модель в файл
                log("Saving model to brain.model")
                save_model(self.model, "brain.model")

        except Exception as e:
            log(f"Error during bot initialization: {e}", level="error")
            raise

    def run(self):
        while True:
            try:
                for symbol in self.symbols:
                    log(f"Fetching latest data for {symbol}")
                    latest_data = self.data_fetcher.fetch_ohlcv(symbol, self.timeframe, limit=100)
                    if latest_data.empty:
                        log(f"No new data for {symbol}. Skipping.", level="warning")
                        continue

                    log(f"Creating features for {symbol}")
                    latest_data = self.feature_engineer.create_features(latest_data)
                    latest_data["symbol"] = symbol

                    required_columns = ['momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 'volume_profile', 'atr', 'upper_band', 'lower_band', 'adx', 'obv']
                    if not all(col in latest_data.columns for col in required_columns):
                        log(f"Missing required columns for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[required_columns].isnull().any().any():
                        log(f"NaN values found in data for {symbol}. Skipping.", level="warning")
                        continue

                    if latest_data[required_columns].empty:
                        log(f"No data available for prediction for {symbol}. Skipping.", level="warning")
                        continue

                    # Получаем сигнал от модели
                    X = latest_data[required_columns].values
                    signal = self.model.predict(X)[0]
                    signal = -1 if signal == 0 else 1

                    entry_price = latest_data["close"].iloc[0]
                    stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, latest_data["atr"].iloc[0])
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
                time.sleep(60)

    def backtest(self):
        return self.backtester.run(self.all_data)

    def generate_explanation(self, latest_data, signal, entry_price, stop_loss, take_profit, position_size):
        explanation = "Decision based on the following indicators:\n"
        explanation += f"- Momentum: {latest_data['momentum'].iloc[0]:.4f} (trend)\n"
        explanation += f"- Volatility: {latest_data['volatility'].iloc[0]:.4f} (risk)\n"
        explanation += f"- RSI: {latest_data['rsi'].iloc[0]:.2f} (overbought/oversold)\n"
        explanation += f"- EMA 20: {latest_data['ema_20'].iloc[0]:.2f} (mid-term trend)\n"
        explanation += f"- MACD: {latest_data['macd'].iloc[0]:.4f}, Signal: {latest_data['macd_signal'].iloc[0]:.4f} (trend dynamics)\n"
        explanation += f"- Volume Profile: {latest_data['volume_profile'].iloc[0]:.2f} (trading activity)\n"
        explanation += f"- ATR: {latest_data['atr'].iloc[0]:.2f} (volatility)\n"
        explanation += f"- Bollinger Bands: Upper {latest_data['upper_band'].iloc[0]:.2f}, Lower {latest_data['lower_band'].iloc[0]:.2f} (volatility boundaries)\n"
        explanation += f"- ADX: {latest_data['adx'].iloc[0]:.2f} (trend strength)\n"
        explanation += f"- OBV: {latest_data['obv'].iloc[0]:.2f} (volume flow)\n"
        explanation += "Signal: BUY (price expected to rise)\n" if signal == 1 else "Signal: SELL (price expected to fall)\n"
        explanation += f"Entry price: {entry_price:.2f}\n"
        explanation += f"Stop loss: {stop_loss:.2f}\n"
        explanation += f"Take profit: {take_profit:.2f}\n"
        explanation += f"Position size: {position_size:.4f} {latest_data['symbol'].iloc[0].split('/')[0]}\n"
        return explanation
