import logging
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf

# Глобальная переменная для уровня логирования
log_level = 3  # 0 - нет логов, 1 - ошибки, 2 - предупреждения и ошибки, 3 - все логи


def prepare_lstm_data(data, sequence_length=50):
    """
    Подготавливает данные для LSTM.
    :param data: DataFrame с OHLCV-данными.
    :param sequence_length: Длина последовательности для LSTM.
    :return: X, y (признаки и целевая переменная).
    """
    # Нормализуем данные
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])

    # Создаем последовательности
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])  # Целевая переменная - цена закрытия

    X, y = np.array(X), np.array(y)
    return X, y, scaler
    
    
def setup_logging():
    """
    Настраивает логирование в зависимости от уровня log_level.
    """
    if log_level == 0:
        logging.basicConfig(level=logging.CRITICAL + 1)  # Отключаем логирование
    elif log_level == 1:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    elif log_level == 2:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    elif log_level == 3:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
def log(message, level="info"):
    """
    Логирует сообщение с указанным уровнем.
    """
    logger = logging.getLogger(__name__)
    if level == "info" and log_level >= 3:
        logger.info(message)
    elif level == "warning" and log_level >= 2:
        logger.warning(message)
    elif level == "error" and log_level >= 1:
        logger.error(message)

def save_model(model, filename, data=None, lstm_model=None):
    """
    Сохраняет модель и данные в файл.
    :param model: Объект модели TradingModel.
    :param filename: Имя файла для сохранения.
    :param data: Данные (например, self.all_data).
    :param lstm_model: Модель LSTM.
    """
    try:
        with open(filename, "wb") as f:
            saved_data = {
                "model": model,
                "data": data,
                "lstm_model": lstm_model
            }
            pickle.dump(saved_data, f)
        log(f"Model and data saved to {filename}", level="info")
    except Exception as e:
        log(f"Error saving model to {filename}: {e}", level="error")

def load_model(filename):
    """
    Загружает модель и данные из файла.
    :param filename: Имя файла для загрузки.
    :return: Модель, данные и модель LSTM.
    """
    try:
        with open(filename, "rb") as f:
            saved_data = pickle.load(f)
            model = saved_data.get("model")
            data = saved_data.get("data")
            lstm_model = saved_data.get("lstm_model")
            log(f"Model and data loaded from {filename}", level="info")
            return model, data, lstm_model
    except Exception as e:
        log(f"Error loading model from {filename}: {e}", level="error")
        return None, None, None
        
def validate_data(data):
    """
    Проверяет данные на корректность (отсутствие NaN и корректные временные метки).
    :param data: DataFrame с данными.
    :return: True, если данные корректны, иначе False.
    """
    if data.empty:
        log("Data is empty.", level="warning")
        return False

    if data.isnull().any().any():
        log("Data contains NaN values.", level="warning")
        return False

    if pd.isna(data.index.min()):
        log("Invalid timestamps in data.", level="warning")
        return False

    return True
    
def convert_timestamps(data):
    """
    Преобразует временные метки в формат datetime.
    :param data: DataFrame с данными.
    :return: DataFrame с преобразованными временными метками.
    """
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
    return data
    
def log_data_range(data):
    """
    Логирует временной промежуток данных.
    :param data: DataFrame с данными.
    """
    if validate_data(data):
        start_date = data.index.min().strftime('%Y-%m-%d %H:%M:%S')
        end_date = data.index.max().strftime('%Y-%m-%d %H:%M:%S')
        log(f"Data range: {start_date} to {end_date}")
        
def save_data(data, filename):
    """
    Сохраняет данные в файл.
    :param data: DataFrame с данными.
    :param filename: Имя файла для сохранения.
    """
    try:
        data.to_pickle(filename)
        log(f"Data saved to {filename}", level="info")
    except Exception as e:
        log(f"Error saving data to {filename}: {e}", level="error")

def load_data(filename):
    """
    Загружает данные из файла.
    :param filename: Имя файла для загрузки.
    :return: DataFrame с данными.
    """
    try:
        data = pd.read_pickle(filename)
        log(f"Data loaded from {filename}", level="info")
        return data
    except Exception as e:
        log(f"Error loading data from {filename}: {e}", level="error")
        return pd.DataFrame()