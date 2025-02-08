import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
from utils import log

class LSTMModel:
    def __init__(self, input_shape):
        """
        Инициализация модели LSTM.
        :param input_shape: Форма входных данных (последовательность, признаки).
        """
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))  # Используем Input(shape) вместо передачи input_shape в LSTM
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))  # Выходной слой для прогнозирования цены

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
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        log("LSTM model trained.")

    def predict(self, X):
        """
        Прогнозирует цену закрытия.
        :param X: Входные данные.
        :return: Прогнозируемая цена.
        """
        return self.model.predict(X)
        
    def train_gradually(self, X_train, y_train, X_val, y_val, max_epochs=100, target_accuracy=0.75):
        """
        Обучает модель постепенно, эпоха за эпохой, пока не достигнет целевой точности.
        :param X_train: Признаки для обучения.
        :param y_train: Целевая переменная для обучения.
        :param X_val: Признаки для валидации.
        :param y_val: Целевая переменная для валидации.
        :param max_epochs: Максимальное количество эпох.
        :param target_accuracy: Целевая точность на валидационных данных.
        :return: Количество выполненных эпох.
        """
        for epoch in range(max_epochs):
            log(f"Training epoch {epoch + 1}/{max_epochs}")
            self.model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

            # Проверяем точность на валидационных данных
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            log(f"Validation accuracy after epoch {epoch + 1}: {val_accuracy:.2%}")

            # Если точность достигнута, останавливаем обучение
            if val_accuracy >= target_accuracy:
                log(f"Target accuracy reached. Stopping training.")
                return epoch + 1

        log(f"Maximum epochs reached. Stopping training.")
        return max_epochs