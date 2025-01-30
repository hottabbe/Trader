import logging
import pickle

def setup_logging(create_logs=True):
    """
    Настраивает логирование.
    """
    if create_logs:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

def log(message, level="info"):
    """
    Логирует сообщение с указанным уровнем.
    """
    logger = logging.getLogger(__name__)
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)

def save_model(model, filename):
    """
    Сохраняет модель в файл.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Загружает модель из файла.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
