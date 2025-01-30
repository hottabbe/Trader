import ccxt
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, news_api_key, log_level=logging.INFO):
        self.exchange = ccxt.mexc({'rateLimit': 1200, 'enableRateLimit': True})
        self.news_api_key = news_api_key
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        try:
            self.logger.info(f"Fetching OHLCV data for {symbol} with timeframe {timeframe}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            self.logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_historical_data(self, symbol, timeframe):
        self.logger.info(f"Fetching all historical data for {symbol}")
        all_data = pd.DataFrame()
        since = None  # Начинаем с текущего момента

        while True:
            # Запрашиваем данные порциями
            data = self.fetch_ohlcv(symbol, timeframe, since=since)
            if data.empty:
                break  # Если данных больше нет, выходим из цикла

            # Добавляем данные в общий DataFrame
            all_data = pd.concat([data, all_data], ignore_index=True)

            # Обновляем начальную дату для следующего запроса
            since = int((data['timestamp'].min() - timedelta(milliseconds=1)).timestamp() * 1000)

            # Если достигли даты листинга пары, выходим из цикла
            if len(data) < 500:
                break

        self.logger.info(f"Total historical data fetched for {symbol}: {len(all_data)} rows")
        return all_data

    def fetch_news(self, query='Bitcoin', language='en', sort_by='publishedAt', page_size=5):
        try:
            self.logger.info(f"Fetching news for query: {query}")
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'apiKey': self.news_api_key
            }
            response = requests.get('https://newsapi.org/v2/everything', params=params)
            if response.status_code == 200:
                self.logger.info(f"Successfully fetched {page_size} news articles")
                return response.json()['articles']
            else:
                self.logger.error(f"Error fetching news: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
