import pandas as pd
import tulipy as ti
import numpy as np
from utils import log

class FeatureEngineer:
    def create_features(self, data):
        """
        Создает технические индикаторы на основе OHLCV-данных.
        """
        if data.empty:
            log("Input data is empty. Skipping feature creation.", level="warning")
            return data

        # Логируем количество строк перед обработкой
        log(f"Creating features for {len(data)} rows")

        # Минимальное количество строк для расчета индикаторов
        min_rows = 50
        if len(data) < min_rows:
            log(f"Not enough data for indicators (min {min_rows} rows required). Skipping.", level="warning")
            return data

        # Рассчитываем технические индикаторы
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Добавляем индикаторы с проверкой длины
        try:
            # SMA
            sma_50 = ti.sma(close, period=50)
            data['sma_50'] = np.nan  # Инициализируем колонку NaN
            data.iloc[-len(sma_50):, data.columns.get_loc('sma_50')] = sma_50  # Заполняем только валидные значения

            sma_200 = ti.sma(close, period=200)
            data['sma_200'] = np.nan
            data.iloc[-len(sma_200):, data.columns.get_loc('sma_200')] = sma_200

            # EMA
            ema_20 = ti.ema(close, period=20)
            data['ema_20'] = np.nan
            data.iloc[-len(ema_20):, data.columns.get_loc('ema_20')] = ema_20

            # RSI
            rsi = ti.rsi(close, period=14)
            data['rsi'] = np.nan
            data.iloc[-len(rsi):, data.columns.get_loc('rsi')] = rsi

            # MACD
            macd, macd_signal, _ = ti.macd(close, short_period=12, long_period=26, signal_period=9)
            data['macd'] = np.nan
            data.iloc[-len(macd):, data.columns.get_loc('macd')] = macd
            data['macd_signal'] = np.nan
            data.iloc[-len(macd_signal):, data.columns.get_loc('macd_signal')] = macd_signal

            # ATR
            atr = ti.atr(high, low, close, period=14)
            data['atr'] = np.nan
            data.iloc[-len(atr):, data.columns.get_loc('atr')] = atr

            # Bollinger Bands
            upper_band, middle_band, lower_band = ti.bbands(close, period=20, stddev=2)
            data['upper_band'] = np.nan
            data.iloc[-len(upper_band):, data.columns.get_loc('upper_band')] = upper_band
            data['middle_band'] = np.nan
            data.iloc[-len(middle_band):, data.columns.get_loc('middle_band')] = middle_band
            data['lower_band'] = np.nan
            data.iloc[-len(lower_band):, data.columns.get_loc('lower_band')] = lower_band

            # ADX
            adx = ti.adx(high, low, close, period=14)
            data['adx'] = np.nan
            data.iloc[-len(adx):, data.columns.get_loc('adx')] = adx

            # OBV
            obv = ti.obv(close, volume)
            data['obv'] = np.nan
            data.iloc[-len(obv):, data.columns.get_loc('obv')] = obv

        except Exception as e:
            log(f"Error calculating indicators: {e}", level="error")
            return data

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