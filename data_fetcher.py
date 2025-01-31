import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
import time  # Добавляем импорт модуля time
from utils import log

class DataFetcher:
    def __init__(self, news_api_key):
        self.exchange = ccxt.mexc({'rateLimit': 1200, 'enableRateLimit': True})
        self.news_api_key = news_api_key

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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Добавляем информацию о временном промежутке
            start_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            end_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            log(f"Successfully fetched {len(df)} rows for {symbol} from {start_date} to {end_date}")

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
            since = int((time.time() - (limit + offset) * self.timeframe_to_seconds(timeframe)) * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            log(f"Error fetching OHLCV data for {symbol}: {e}", level="error")
            return pd.DataFrame()

    def fetch_all_historical_data(self, symbol, timeframe, max_iterations=11):
        """
        Получает все исторические данные для указанного символа и таймфрейма.
        """
        log(f"Fetching all historical data for {symbol}")
        all_data = pd.DataFrame()
        since = None  # Начинаем с текущего момента
        total_rows = 0
        iteration = 0

        while iteration < max_iterations:
            # Запрашиваем данные порциями
            data = self.fetch_ohlcv(symbol, timeframe, since=since)
            if data.empty:
                break  # Если данных больше нет, выходим из цикла

            # Логируем общее количество строк и временной промежуток
            start_date = data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            end_date = data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {symbol}: {len(all_data) + len(data)} rows ({start_date} - {end_date})")

            # Убедимся, что индексы уникальны и сброшены
            data.reset_index(drop=True, inplace=True)

            # Добавляем данные в общий DataFrame
            all_data = pd.concat([all_data, data], ignore_index=True)
            total_rows += len(data)

            # Обновляем начальную дату для следующего запроса
            since = int((data['timestamp'].min() - timedelta(milliseconds=1)).timestamp() * 1000)
            iteration += 1

            # Если достигли даты листинга пары, выходим из цикла
            if len(data) < 500:
                break

        log(f"Total historical data fetched for {symbol}: {total_rows} rows")
        return all_data

    def fetch_news(self, query='Bitcoin', language='en', sort_by='publishedAt', page_size=5):
        """
        Получает новости, связанные с криптовалютой.
        """
        try:
            log(f"Fetching news for query: {query}")
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'apiKey': self.news_api_key
            }
            response = requests.get('https://newsapi.org/v2/everything', params=params)
            if response.status_code == 200:
                log(f"Successfully fetched {page_size} news articles")
                return response.json()['articles']
            else:
                log(f"Error fetching news: {response.status_code}", level="error")
                return []
        except Exception as e:
            log(f"Error fetching news: {e}", level="error")
            return []