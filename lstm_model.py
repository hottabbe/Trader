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

    def train(self, X_train, y_train, epochs=50, batch_size=32):
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