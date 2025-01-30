import ccxt
import requests
import polars as pl
from utils import log
import os

class DataFetcher:
    def __init__(self, news_api_key):
        self.exchange = ccxt.mexc({'rateLimit': 1200, 'enableRateLimit': True})
        self.news_api_key = news_api_key

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        try:
            log(f"Fetching OHLCV data for {symbol} with timeframe {timeframe}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                log(f"No data returned for {symbol}", level="warning")
                return pl.DataFrame()

            df = pl.DataFrame(ohlcv, schema=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.with_column(pl.col("timestamp").apply(lambda x: datetime.fromtimestamp(x / 1000)).alias("timestamp"))
            log(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            log(f"Error fetching OHLCV data for {symbol}: {e}", level="error")
            return pl.DataFrame()

    def fetch_all_historical_data(self, symbol, timeframe):
        cache_file = f"{symbol}_{timeframe}_data.parquet"
        if os.path.exists(cache_file):
            log(f"Loading cached data for {symbol} from {cache_file}")
            return pl.read_parquet(cache_file)
        else:
            open(cache_file, "a").close()
        log(f"Fetching all historical data for {symbol}")
        all_data = pl.DataFrame()
        since = None  # Начинаем с текущего момента

        while True:
            data = self.fetch_ohlcv(symbol, timeframe, since=since)
            if data.is_empty():
                break  # Если данных больше нет, выходим из цикла

            all_data = pl.concat([all_data, data])
            since = int((data["timestamp"].min().timestamp() - 1) * 1000)  # Обновляем начальную дату

            if len(data) < 500:
                break  # Если данных меньше 500, значит, мы достигли конца

        log(f"Total historical data fetched for {symbol}: {len(all_data)} rows")
        all_data.write_parquet(cache_file)  # Кэшируем данные
        return all_data

    def fetch_news(self, query='Bitcoin', language='en', sort_by='publishedAt', page_size=5):
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