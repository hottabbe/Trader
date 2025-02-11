import ccxt
import pandas as pd
from datetime import datetime
from utils import log

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()  # Используем Binance API для актуальных данных

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        """
        Получает актуальные OHLCV-данные для указанного символа и таймфрейма через API Binance.
        :param symbol: Торговая пара (например, 'BTC/USDT').
        :param timeframe: Таймфрейм (например, '1h').
        :param since: Временная метка для начала загрузки данных (опционально).
        :param limit: Количество строк данных.
        :return: DataFrame с данными.
        """
        try:
            log(f"Fetching OHLCV data for {symbol} with timeframe {timeframe}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                log(f"No data returned for {symbol}", level="warning")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Преобразуем временные метки
            return df
        except Exception as e:
            log(f"Error fetching OHLCV data for {symbol}: {e}", level="error")
            return pd.DataFrame()