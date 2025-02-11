import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from utils import log

class LSTMModel:
    def __init__(self, input_shape):
        """
        Инициализация модели LSTM.
        :param input_shape: Форма входных данных (последовательность, признаки).
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)  # Выходной слой для прогнозирования цены
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        log("LSTM model initialized.")

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """
        Обучает модель LSTM.
        :param X_train: Признаки для обучения.
        :param y_train: Целевая переменная для обучения.
        :param epochs: Количество эпох.
        :param batch_size: Размер батча.
        """
        if X_train.size == 0 or y_train.size == 0:
            log("Training data is empty. Skipping training.", level="warning")
            return

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        log("LSTM model trained.")

    def predict(self, X):
        """
        Прогнозирует цену закрытия.
        :param X: Входные данные.
        :return: Прогнозируемая цена.
        """
        if X.size == 0:
            log("Prediction data is empty. Skipping prediction.", level="warning")
            return None

        return self.model.predict(X)

    def train_gradually(self, X_train, y_train, X_val, y_val, max_epochs=100, target_loss=0.01):
        """
        Обучает модель постепенно, эпоха за эпохой, пока не достигнет целевого значения потерь.
        :param X_train: Признаки для обучения.
        :param y_train: Целевая переменная для обучения.
        :param X_val: Признаки для валидации.
        :param y_val: Целевая переменная для валидации.
        :param max_epochs: Максимальное количество эпох.
        :param target_loss: Целевое значение потерь на валидационных данных.
        :return: Количество выполненных эпох.
        """
        for epoch in range(max_epochs):
            log(f"Training epoch {epoch + 1}/{max_epochs}")
            self.model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

            # Проверяем потери на валидационных данных
            val_loss = self.model.evaluate(X_val, y_val, verbose=1)
            log(f"Validation loss after epoch {epoch + 1}: {val_loss:.4f}")

            # Если потери достигнуты, останавливаем обучение
            if val_loss <= target_loss:
                log(f"Target loss reached. Stopping training.")
                return epoch + 1

        log(f"Maximum epochs reached. Stopping training.")
        return max_epochs