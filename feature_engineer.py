import pandas as pd
import tulipy as ti
import numpy as np

class FeatureEngineer:
    def create_features(self, data):
        """
        Создает технические индикаторы на основе OHLCV-данных с использованием tulipy.
        """
        if data.empty:
            return data

        # Рассчитываем технические индикаторы
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Примеры индикаторов
        data["rsi"] = ti.rsi(close, period=14)
        data["ema_20"] = ti.ema(close, period=20)
        data["macd"], data["macd_signal"], _ = ti.macd(close, short_period=12, long_period=26, signal_period=9)
        data["atr"] = ti.atr(high, low, close, period=14)
        data["upper_band"], data["middle_band"], data["lower_band"] = ti.bbands(close, period=20, stddev=2)
        data["adx"] = ti.adx(high, low, close, period=14)
        data["obv"] = ti.obv(close, volume)

        # Добавляем пользовательские индикаторы
        data["return"] = data["close"].pct_change()
        data["momentum"] = data["close"].pct_change(4)
        data["volatility"] = data["close"].pct_change().rolling(window=14).std()

        # Удаляем строки с NaN (из-за расчета индикаторов)
        data.dropna(inplace=True)

        return data
