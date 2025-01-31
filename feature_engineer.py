import pandas as pd
import tulipy as ti
import numpy as np
from utils import log

class FeatureEngineer:
    def create_features(self, data):
        """
        Создает технические индикаторы на основе OHLCV-данных с использованием tulipy.
        """
        if data.empty:
            return data

        # Логируем количество строк, индексов и значений перед обработкой
        log(f"Creating features for {len(data)} rows, {len(data.index)} indices, {len(data.values)} values")

        # Рассчитываем технические индикаторы
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Примеры индикаторов
        data["rsi"] = np.nan
        data["ema_20"] = np.nan
        data["macd"] = np.nan
        data["macd_signal"] = np.nan
        data["atr"] = np.nan
        data["upper_band"] = np.nan
        data["middle_band"] = np.nan
        data["lower_band"] = np.nan
        data["adx"] = np.nan
        data["obv"] = np.nan

        # Рассчитываем индикаторы и сдвигаем их, чтобы длина совпадала
        if len(close) >= 14:
            rsi = ti.rsi(close, period=14)
            data.loc[len(data) - len(rsi):, "rsi"] = rsi  # Сдвигаем RSI
        if len(close) >= 20:
            ema_20 = ti.ema(close, period=20)
            data.loc[len(data) - len(ema_20):, "ema_20"] = ema_20  # Сдвигаем EMA
        if len(close) >= 26:
            macd, macd_signal, _ = ti.macd(close, short_period=12, long_period=26, signal_period=9)
            data.loc[len(data) - len(macd):, "macd"] = macd  # Сдвигаем MACD
            data.loc[len(data) - len(macd_signal):, "macd_signal"] = macd_signal  # Сдвигаем MACD Signal
        if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
            atr = ti.atr(high, low, close, period=14)
            data.loc[len(data) - len(atr):, "atr"] = atr  # Сдвигаем ATR
        if len(close) >= 20:
            upper_band, middle_band, lower_band = ti.bbands(close, period=20, stddev=2)
            data.loc[len(data) - len(upper_band):, "upper_band"] = upper_band  # Сдвигаем Bollinger Bands
            data.loc[len(data) - len(middle_band):, "middle_band"] = middle_band
            data.loc[len(data) - len(lower_band):, "lower_band"] = lower_band
        if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
            adx = ti.adx(high, low, close, period=14)
            data.loc[len(data) - len(adx):, "adx"] = adx  # Сдвигаем ADX
        if len(close) >= 1 and len(volume) >= 1:
            obv = ti.obv(close, volume)
            data.loc[len(data) - len(obv):, "obv"] = obv  # Сдвигаем OBV

        # Добавляем пользовательские индикаторы
        data["return"] = data["close"].pct_change()
        data["momentum"] = data["close"].pct_change(4)
        data["volatility"] = data["close"].pct_change().rolling(window=14).std()

        # Добавляем volume_profile (скользящее среднее объема)
        data["volume_profile"] = data["volume"].rolling(window=14).mean()

        # Удаляем строки с NaN (из-за расчета индикаторов)
        data.dropna(inplace=True)

        # Логируем количество строк, индексов и значений после обработки
        log(f"Features created for {len(data)} rows, {len(data.index)} indices, {len(data.values)} values")

        return data