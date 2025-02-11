import logging
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime

# Глобальная переменная для уровня логирования
log_level = 3  # 0 - нет логов, 1 - ошибки, 2 - предупреждения и ошибки, 3 - все логи

def setup_logging():
    """
    Настраивает логирование в зависимости от уровня log_level.
    """
    if not logging.getLogger().hasHandlers():  # Проверяем, не настроено ли логирование уже
        _format = "%(asctime)s |•| %(levelname)s |•| %(message)s"
        dformat = "%d/%b/%y %H:%M:%S"
        if log_level == 0:
            logging.basicConfig(level=logging.CRITICAL + 1)  # Отключаем логирование
        elif log_level == 1:
            logging.basicConfig(level=logging.ERROR, format=_format, datefmt=dformat)
        elif log_level == 2:
            logging.basicConfig(level=logging.WARNING, format=_format, datefmt=dformat)
        elif log_level == 3:
            logging.basicConfig(level=logging.INFO, format=_format, datefmt=dformat)

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
    Сохраняет модель и данные в файл в папке data.
    :param model: Модель для сохранения.
    :param filename: Имя файла.
    :param data: Данные для сохранения (опционально).
    :param lstm_model: LSTM-модель для сохранения (опционально).
    """
    try:
        os.makedirs("data", exist_ok=True)  # Создаем папку data, если её нет
        filepath = os.path.join("data", filename)
        with open(filepath, "wb") as f:
            saved_data = {
                "model": model,
                "data": data,
                "lstm_model": lstm_model
            }
            pickle.dump(saved_data, f)
        log(f"Model and data saved to {filepath}", level="info")
    except Exception as e:
        log(f"Error saving model to {filepath}: {e}", level="error")

def load_model(filename):
    """
    Загружает модель и данные из файла в папке data.
    :param filename: Имя файла.
    :return: Кортеж (model, data, lstm_model), где только один элемент не None.
    """
    try:
        filepath = os.path.join("data", filename)
        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)
            model = saved_data.get("model")
            data = saved_data.get("data")
            lstm_model = saved_data.get("lstm_model")

            # Возвращаем только непустые элементы
            if model is not None:
                return model, None, None
            elif data is not None:
                return None, data, None
            elif lstm_model is not None:
                return None, None, lstm_model
            else:
                log(f"No valid data found in {filepath}.", level="warning")
                return None, None, None

    except Exception as e:
        log(f"Error loading model from {filepath}: {e}", level="error")
        return None, None, None

def validate_data(data):
    """
    Проверяет данные на корректность (отсутствие NaN и корректные временные метки).
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
    """
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
    return data

def log_data_range(data):
    """
    Логирует временной промежуток данных.
    Предполагается, что data.index содержит временные метки.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index, unit='s')
        except Exception as e:
            raise ValueError(f"Не удалось преобразовать индекс в datetime: {e}")

    # Получите минимальную и максимальную даты
    start_date = data.index.min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = data.index.max().strftime('%Y-%m-%d %H:%M:%S')

    # Логирование диапазона дат
    log(f"Data range: {start_date} to {end_date}")

def save_data(data, filename):
    """
    Сохраняет данные в файл в папке data.
    """
    try:
        os.makedirs("data", exist_ok=True)  # Создаем папку data, если её нет
        filepath = os.path.join("data", filename)
        data.to_pickle(filepath)
        log(f"Data saved to {filepath}", level="info")
    except Exception as e:
        log(f"Error saving data to {filepath}: {e}", level="error")

def load_data(filename):
    """
    Загружает данные из файла в папке data.
    """
    try:
        filepath = os.path.join("data", filename)
        data = pd.read_pickle(filepath)
        log(f"Data loaded from {filepath}", level="info")
        return data
    except Exception as e:
        log(f"Error loading data from {filepath}: {e}", level="error")
        return pd.DataFrame()

def fetch_all_historical_data(symbol, timeframe):
    """
    Загружает все исторические данные для указанного символа и таймфрейма из архивов Binance.
    :param symbol: Торговая пара (например, 'BTCUSDT').
    :param timeframe: Таймфрейм (например, '1h').
    :return: DataFrame с историческими данными.
    """
    log(f"Fetching all historical data for {symbol} from Binance archives")

    # Определяем дату листинга пары
    listing_date = find_listing_date(symbol)
    if not listing_date:
        log(f"Could not find listing date for {symbol}. Skipping.", level="warning")
        return pd.DataFrame()

    # Определяем имя файла для сохранения данных
    end_date = datetime.utcnow()
    data_filename = f"{symbol}_{timeframe}_{listing_date.strftime('%Y-%m')}_{end_date.strftime('%Y-%m')}.pkl"

    # Если файл уже существует, загружаем данные из него
    if os.path.exists(os.path.join("data", data_filename)):
        log(f"Loading historical data for {symbol} from {data_filename}")
        return load_data(data_filename)

    all_data = pd.DataFrame()
    archive_base_url = "https://data.binance.vision/data/spot/monthly/klines"

    # Загружаем данные по месяцам
    current_date = listing_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month

        # Формируем URL для загрузки архива
        archive_url = f"{archive_base_url}/{symbol.replace('/', '')}/{timeframe}/{symbol.replace('/', '')}-{timeframe}-{year}-{month:02d}.zip"
        log(f"Downloading data for {symbol} from {archive_url}")

        try:
            # Загружаем архив
            response = requests.get(archive_url)
            response.raise_for_status()

            # Распаковываем архив
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Получаем имя файла внутри архива
                csv_filename = zip_file.namelist()[0]
                with zip_file.open(csv_filename) as csv_file:
                    # Читаем CSV-файл
                    data = pd.read_csv(
                        csv_file,
                        names=[
                            "timestamp", "open", "high", "low", "close", "volume",
                            "close_time", "quote_asset_volume", "number_of_trades",
                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                        ]
                    )
                    # Оставляем только нужные колонки
                    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
                    # Преобразуем временные метки
                    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
                    # Добавляем данные в общий DataFrame
                    all_data = pd.concat([all_data, data], ignore_index=True)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                log(f"Data for {symbol} {year}-{month:02d} not found on Binance. Skipping.", level="warning")
            else:
                log(f"Error downloading or processing data for {symbol} {year}-{month:02d}: {e}", level="warning")
        except Exception as e:
            log(f"Error downloading or processing data for {symbol} {year}-{month:02d}: {e}", level="warning")

        # Переходим к следующему месяцу
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    # Сохраняем данные в файл
    if not all_data.empty:
        log(f"Saving historical data for {symbol} to {data_filename}")
        save_data(all_data, data_filename)

    return all_data

def find_listing_date(symbol):
    """
    Находит дату листинга пары на Binance.
    :param symbol: Торговая пара (например, 'BTCUSDT').
    :return: Дата листинга или None, если дата не найдена.
    """
    # Примерная дата листинга для BTCUSDT (можно уточнить для других пар)
    if symbol == "BTCUSDT":
        return datetime(2017, 8, 17)
    elif symbol == "ETHUSDT":
        return datetime(2017, 8, 17)
    elif symbol == "BNBUSDT":
        return datetime(2017, 8, 17)
    else:
        log(f"Listing date for {symbol} not found. Using default date.", level="warning")
        return datetime(2017, 8, 17)  # По умолчанию используем дату листинга BTCUSDT

def prepare_lstm_data(data, sequence_length=50):
    """
    Подготавливает данные для LSTM.
    :param data: DataFrame с OHLCV-данными.
    :param sequence_length: Длина последовательности для LSTM.
    :return: X, y (признаки и целевая переменная).
    """
    if data.empty or len(data) < sequence_length:
        log("Not enough data for LSTM. Skipping.", level="warning")
        return np.array([]), np.array([]), None

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