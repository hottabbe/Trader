import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from utils import log

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})

    def timeframe_to_seconds(self, timeframe):
        """
        Конвертирует таймфрейм в секунды.
        :param timeframe: Таймфрейм (например, '1h').
        :return: Количество секунд.
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        """
        Получает OHLCV-данные для указанного символа и таймфрейма.
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

        def fetch_ohlcv_with_offset(self, symbol, timeframe, limit=500, offset=0):
            """
            Получает OHLCV-данные с учетом смещения во времени.
            :param symbol: Торговая пара (например, 'BTC/USDT').
            :param timeframe: Таймфрейм (например, '1h').
            :param limit: Количество строк данных.
            :param offset: Смещение во времени (в количестве свечей).
            :return: DataFrame с данными.
            """
            try:
                since = int((time.time() - (limit + offset) * self.timeframe_to_seconds(timeframe)) * 500)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                log(f"Error fetching OHLCV data for {symbol}: {e}", level="error")
                return pd.DataFrame()

    def fetch_all_historical_data(self, symbol, timeframe):
        """
        Получает все исторические данные для указанного символа и таймфрейма, начиная с даты листинга.
        :param symbol: Торговая пара (например, 'BTC/USDT').
        :param timeframe: Таймфрейм (например, '1h').
        :return: DataFrame с историческими данными.
        """
        log(f"Fetching all historical data for {symbol}")

        # Находим дату листинга
        listing_date = self.find_listing_date(symbol, timeframe)
        if not listing_date:
            log(f"Could not find listing date for {symbol}. Skipping.", level="warning")
            return pd.DataFrame()

        # Получаем данные с даты листинга до текущего момента
        all_data = pd.DataFrame()  # Инициализируем пустой DataFrame
        since = listing_date

        try:
            while True:
                # Запрашиваем данные порциями по 500 свечей
                data = self.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
                if data.empty:
                    break  # Если данных больше нет, выходим из цикла

                # Логируем общее количество строк и временной промежуток
                start_date = data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
                end_date = data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                log(f"{symbol}: {len(all_data) + len(data)} rows ({start_date} - {end_date})")

                # Убедимся, что индексы уникальны и сброшены
                data.reset_index(drop=True, inplace=True)

                # Добавляем данные в общий DataFrame
                all_data = pd.concat([all_data, data], ignore_index=True)

                # Обновляем начальную дату для следующего запроса
                since = int((data['timestamp'].max() + pd.Timedelta(seconds=1)).timestamp() * 1000)

                # Если достигли текущего момента, выходим из цикла
                if len(data) < 500:
                    break

            log(f"Total historical data fetched for {symbol}: {len(all_data)} rows")
            return all_data

        except Exception as e:
            log(f"Error fetching historical data for {symbol}: {e}", level="error")
            return pd.DataFrame()
        
    def find_listing_date(self, symbol, timeframe):
        """
        Находит дату листинга для указанного символа.
        :param symbol: Торговая пара (например, 'BTC/USDT').
        :param timeframe: Таймфрейм (например, '1h').
        :return: Временная метка (timestamp) даты листинга или None, если дата не найдена.
        """
        current_date = self.exchange.parse8601('2015-01-01T00:00:00Z')  # Очень ранняя дата
        step = 86400 * 1000 * 30  # Шаг в миллисекундах (месяц)

        try:
            while True:
                log(f"Checking data starting from {datetime.fromtimestamp(current_date // 1000)}")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_date, limit=1)

                if ohlcv:
                    listing_date = ohlcv[0][0]  # Берём временную метку первой записи
                    log(f"Found first available record: {datetime.fromtimestamp(listing_date // 1000)}")
                    return listing_date

                # Если данных нет, увеличиваем временной диапазон
                current_date += step

                # Ограничение по времени (не будем проверять дальше 2023 года)
                if current_date > self.exchange.parse8601('2023-01-01T00:00:00Z'):
                    log("Exceeded maximum search time.", level="warning")
                    return None

        except Exception as e:
            log(f"Error finding listing date: {e}", level="error")
            return None