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
            log("Input data is empty. Skipping feature creation.", level="warning")
            return data

        # Логируем количество строк перед обработкой
        log(f"Creating features for {len(data)} rows")

        # Минимальное количество строк для расчета индикаторов
        min_rows = 50  # Например, RSI требует минимум 14 строк
        if len(data) < min_rows:
            log(f"Not enough data for indicators (min {min_rows} rows required). Skipping.", level="warning")
            return data

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
            data.iloc[-len(rsi):, data.columns.get_loc('rsi')] = rsi

        if len(close) >= 20:
            ema_20 = ti.ema(close, period=20)
            data.iloc[-len(ema_20):, data.columns.get_loc('ema_20')] = ema_20

        if len(close) >= 26:
            macd, macd_signal, _ = ti.macd(close, short_period=12, long_period=26, signal_period=9)
            data.iloc[-len(macd):, data.columns.get_loc('macd')] = macd
            data.iloc[-len(macd_signal):, data.columns.get_loc('macd_signal')] = macd_signal

        if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
            atr = ti.atr(high, low, close, period=14)
            data.iloc[-len(atr):, data.columns.get_loc('atr')] = atr

        if len(close) >= 20:
            upper_band, middle_band, lower_band = ti.bbands(close, period=20, stddev=2)
            data.iloc[-len(upper_band):, data.columns.get_loc('upper_band')] = upper_band
            data.iloc[-len(middle_band):, data.columns.get_loc('middle_band')] = middle_band
            data.iloc[-len(lower_band):, data.columns.get_loc('lower_band')] = lower_band

        if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
            adx = ti.adx(high, low, close, period=14)
            data.iloc[-len(adx):, data.columns.get_loc('adx')] = adx

        if len(close) >= 1 and len(volume) >= 1:
            obv = ti.obv(close, volume)
            data.iloc[-len(obv):, data.columns.get_loc('obv')] = obv

        # Добавляем пользовательские индикаторы
        data["return"] = data["close"].pct_change()
        data["momentum"] = data["close"].pct_change(4)
        data["volatility"] = data["close"].pct_change().rolling(window=14).std()

        # Добавляем volume_profile (скользящее среднее объема)
        data["volume_profile"] = data["volume"].rolling(window=14).mean()

        # Логируем количество строк перед удалением NaN
        log(f"Rows before dropping NaN: {len(data)}")

        # Удаляем строки с NaN (из-за расчета индикаторов)
        data.dropna(inplace=True)

        # Логируем количество строк после удаления NaN
        log(f"Rows after dropping NaN: {len(data)}")

        if data.empty:
            log("Data is empty after dropping NaN. Check your indicators.", level="error")
            return data

        # Логируем количество строк после обработки
        log(f"Features created for {len(data)} rows")

        return data