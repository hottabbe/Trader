import polars as pl
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self, create_logs=True):
        self.create_logs = create_logs
        if create_logs:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()

    @staticmethod
    def compute_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    @staticmethod
    def compute_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_atr(high, low, close, period=14):
        tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def compute_bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    @staticmethod
    def compute_ichimoku(high, low, close):
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    @staticmethod
    def compute_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def compute_obv(close, volume):
        obv = (volume * np.sign(close.diff())).cumsum()
        return obv

    def create_features(self, df):
        if self.create_logs:
            self.logger.info("Creating features for the dataset")

        df = df.with_column(
            (pl.col("close").pct_change().alias("return"))
        ).with_column(
            (pl.col("return").rolling_mean(5).alias("momentum"))
        ).with_column(
            (pl.col("return").rolling_std(5).alias("volatility"))
        ).with_column(
            (self.compute_rsi(pl.col("close"), 14).alias("rsi"))
        ).with_column(
            (self.compute_ema(pl.col("close"), 20).alias("ema_20"))
        )

        macd, signal = self.compute_macd(pl.col("close"))
        df = df.with_column(macd.alias("macd")).with_column(signal.alias("macd_signal"))

        df = df.with_column(
            (pl.col("volume").rolling_mean(20).alias("volume_profile"))
        ).with_column(
            (self.compute_atr(pl.col("high"), pl.col("low"), pl.col("close")).alias("atr"))
        )

        upper_band, lower_band = self.compute_bollinger_bands(pl.col("close"))
        df = df.with_column(upper_band.alias("upper_band")).with_column(lower_band.alias("lower_band"))

        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = self.compute_ichimoku(pl.col("high"), pl.col("low"), pl.col("close"))
        df = df.with_column(tenkan_sen.alias("tenkan_sen")).with_column(kijun_sen.alias("kijun_sen"))
        df = df.with_column(senkou_span_a.alias("senkou_span_a")).with_column(senkou_span_b.alias("senkou_span_b"))

        adx = self.compute_adx(pl.col("high"), pl.col("low"), pl.col("close"))
        df = df.with_column(adx.alias("adx"))

        obv = self.compute_obv(pl.col("close"), pl.col("volume"))
        df = df.with_column(obv.alias("obv"))

        # Нормализация данных
        numeric_columns = ['momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 'volume_profile', 'atr', 'upper_band', 'lower_band', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'adx', 'obv']
        df = df.with_column(
            pl.struct(numeric_columns).apply(lambda x: self.scaler.fit_transform(x.to_numpy().reshape(-1, 1)).flatten()).alias("normalized_features")
        )

        df = df.drop_nulls()  # Удаляем строки с NaN значениями

        if self.create_logs:
            self.logger.info("Features created successfully")
        return df