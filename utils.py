import logging
import os
import pickle
from datetime import datetime

# Глобальная переменная для управления логированием
CREATE_LOGS = True

def setup_logging(create_logs=True):
    """Настройка логирования."""
    global CREATE_LOGS
    CREATE_LOGS = create_logs
    if create_logs:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def log(message, level="info"):
    """Логирует сообщение, если логирование включено."""
    if CREATE_LOGS:
        logger = logging.getLogger(__name__)
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        else:
            logger.info(message)  # По умолчанию используем info

def save_model(model, filename):
    """Сохраняет модель в файл."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        log(f"Model saved to {filename}")
    except Exception as e:
        log(f"Error saving model: {e}", level="error")

def load_model(filename):
    """Загружает модель из файла."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        log(f"Model loaded from {filename}")
        return model
    except Exception as e:
        log(f"Error loading model: {e}", level="error")
        return None

def timestamp_to_datetime(timestamp):
    """Преобразует timestamp в datetime."""
    return datetime.fromtimestamp(timestamp / 1000)

def datetime_to_timestamp(dt):
    """Преобразует datetime в timestamp."""
    return int(dt.timestamp() * 1000)

def ensure_directory_exists(directory):
    """Создает директорию, если она не существует."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}")