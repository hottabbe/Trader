import logging
import pickle
import os

# Глобальная переменная для уровня логирования
log_level = 3  # 0 - нет логов, 1 - ошибки, 2 - предупреждения и ошибки, 3 - все логи

def setup_logging():
    """
    Настраивает логирование в зависимости от уровня log_level.
    """
    if log_level == 0:
        logging.basicConfig(level=logging.CRITICAL + 1)  # Отключаем логирование
    elif log_level == 1:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
    elif log_level == 2:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    elif log_level == 3:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def save_model(model, filename):
    """
    Сохраняет модель в файл.
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        log(f"Model saved to {filename}", level="info")
    except Exception as e:
        log(f"Error saving model to {filename}: {e}", level="error")

def load_model(filename):
    """
    Загружает модель из файла.
    """
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        log(f"Model loaded from {filename}", level="info")
        return model
    except Exception as e:
        log(f"Error loading model from {filename}: {e}", level="error")
        return None