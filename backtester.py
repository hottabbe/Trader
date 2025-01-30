import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class Backtester:
    def __init__(self, model, create_logs=True):
        self.model = model
        self.create_logs = create_logs
        if create_logs:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def run(self, data):
        try:
            if self.create_logs:
                self.logger.info("Running backtest")

            if data.is_empty():
                raise ValueError("No data available for backtesting.")

            # Создаем целевую переменную
            data = data.with_column(
                (data["return"].shift(-1) > 0).cast(np.int8).alias("signal")
            )

            # Разделяем данные на обучающую и тестовую выборки
            train_data = data.slice(0, -100)  # Все данные, кроме последних 100 строк
            test_data = data.slice(-100, 100)  # Последние 100 строк

            if train_data.is_empty() or test_data.is_empty():
                raise ValueError("Not enough data to create train and test sets.")

            if self.create_logs:
                self.logger.info(f"Training data points: {len(train_data)}")
                self.logger.info(f"Test data points: {len(test_data)}")

            # Обучаем модель на исторических данных
            required_columns = ['momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 'volume_profile', 'atr', 'upper_band', 'lower_band', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'adx', 'obv']
            X_train = train_data.select(required_columns).to_numpy()
            y_train = train_data["signal"].to_numpy()

            if self.create_logs:
                self.logger.info("Training the model")
            self.model.train(X_train, y_train)

            # Оцениваем модель на тестовых данных
            X_test = test_data.select(required_columns).to_numpy()
            y_test = test_data["signal"].to_numpy()

            accuracy = accuracy_score(y_test, self.model.predict(X_test))
            precision = precision_score(y_test, self.model.predict(X_test))
            recall = recall_score(y_test, self.model.predict(X_test))

            if self.create_logs:
                self.logger.info(f"Model accuracy on test data: {accuracy:.2%}")
                self.logger.info(f"Model precision: {precision:.2%}")
                self.logger.info(f"Model recall: {recall:.2%}")

            return accuracy, precision, recall

        except Exception as e:
            if self.create_logs:
                self.logger.error(f"Error during backtesting: {e}")
            raise