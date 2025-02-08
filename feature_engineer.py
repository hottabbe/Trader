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
        min_rows = 200  # Увеличили минимальное количество строк для более сложных индикаторов
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
            # Скользящие средние
            data['sma_20'] = np.nan
            sma_20 = ti.sma(close, period=20)
            data.iloc[-len(sma_20):, data.columns.get_loc('sma_20')] = sma_20

            data['sma_50'] = np.nan
            sma_50 = ti.sma(close, period=50)
            data.iloc[-len(sma_50):, data.columns.get_loc('sma_50')] = sma_50

            data['sma_200'] = np.nan
            sma_200 = ti.sma(close, period=200)
            data.iloc[-len(sma_200):, data.columns.get_loc('sma_200')] = sma_200

            # Экспоненциальные скользящие средние
            data['ema_20'] = np.nan
            ema_20 = ti.ema(close, period=20)
            data.iloc[-len(ema_20):, data.columns.get_loc('ema_20')] = ema_20

            data['ema_50'] = np.nan
            ema_50 = ti.ema(close, period=50)
            data.iloc[-len(ema_50):, data.columns.get_loc('ema_50')] = ema_50

            data['ema_200'] = np.nan
            ema_200 = ti.ema(close, period=200)
            data.iloc[-len(ema_200):, data.columns.get_loc('ema_200')] = ema_200

            # Индекс относительной силы (RSI)
            data['rsi_14'] = np.nan
            rsi_14 = ti.rsi(close, period=14)
            data.iloc[-len(rsi_14):, data.columns.get_loc('rsi_14')] = rsi_14

            data['rsi_28'] = np.nan
            rsi_28 = ti.rsi(close, period=28)
            data.iloc[-len(rsi_28):, data.columns.get_loc('rsi_28')] = rsi_28

            # MACD
            data['macd'] = np.nan
            data['macd_signal'] = np.nan
            macd, macd_signal, _ = ti.macd(close, short_period=12, long_period=26, signal_period=9)
            data.iloc[-len(macd):, data.columns.get_loc('macd')] = macd
            data.iloc[-len(macd_signal):, data.columns.get_loc('macd_signal')] = macd_signal

            # ATR (Average True Range)
            data['atr_14'] = np.nan
            atr_14 = ti.atr(high, low, close, period=14)
            data.iloc[-len(atr_14):, data.columns.get_loc('atr_14')] = atr_14

            # Bollinger Bands
            data['upper_band'] = np.nan
            data['middle_band'] = np.nan
            data['lower_band'] = np.nan
            upper_band, middle_band, lower_band = ti.bbands(close, period=20, stddev=2)
            data.iloc[-len(upper_band):, data.columns.get_loc('upper_band')] = upper_band
            data.iloc[-len(middle_band):, data.columns.get_loc('middle_band')] = middle_band
            data.iloc[-len(lower_band):, data.columns.get_loc('lower_band')] = lower_band

            # ADX (Average Directional Index)
            data['adx_14'] = np.nan
            adx_14 = ti.adx(high, low, close, period=14)
            data.iloc[-len(adx_14):, data.columns.get_loc('adx_14')] = adx_14

            # OBV (On-Balance Volume)
            data['obv'] = np.nan
            obv = ti.obv(close, volume)
            data.iloc[-len(obv):, data.columns.get_loc('obv')] = obv

            # CCI (Commodity Channel Index)
            data['cci_20'] = np.nan
            cci_20 = ti.cci(high, low, close, period=20)
            data.iloc[-len(cci_20):, data.columns.get_loc('cci_20')] = cci_20

            # Stochastic Oscillator
            data['stoch_k'] = np.nan
            data['stoch_d'] = np.nan
            stoch_k, stoch_d = ti.stoch(high, low, close, 14, 3, 3)  # Исправленный вызов
            data.iloc[-len(stoch_k):, data.columns.get_loc('stoch_k')] = stoch_k
            data.iloc[-len(stoch_d):, data.columns.get_loc('stoch_d')] = stoch_d

        except Exception as e:
            log(f"Error calculating indicators: {e}", level="error")
            return data

        # Добавляем пользовательские индикаторы
        data["return"] = data["close"].pct_change()
        data["momentum"] = data["close"].pct_change(4)
        data["volatility"] = data["close"].pct_change().rolling(window=14).std()

        # Добавляем volume_profile (скользящее среднее объема)
        data["volume_profile"] = data["volume"].rolling(window=14).mean()

        # Целевая переменная: изменение цены через 5 свечей
        data["target_direction"] = (data["close"].shift(-5) > data["close"]).astype(int)  # 1 если рост, 0 если падение
        data["target_level"] = (data["close"].shift(-5) - data["close"]) / data["close"]  # Процентное изменение

        # Определяем сегменты графика
        data = self.identify_market_segments(data)

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

    def identify_market_segments(self, data, window=50):
        """
        Определяет сегменты графика по характеру движения (тренд, боковик).
        :param data: DataFrame с данными.
        :param window: Размер окна для анализа.
        :return: DataFrame с добавленным столбцом 'segment'.
        """
        # Рассчитываем скользящие средние
        data['sma_50'] = data['close'].rolling(window=window).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()

        # Определяем тренд
        data['trend'] = np.where(data['sma_50'] > data['sma_200'], 'uptrend', 'downtrend')

        # Определяем боковик (флэт)
        data['range'] = data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min()
        data['atr'] = data['atr_14']  # Используем ATR для определения волатильности
        data['is_flat'] = data['range'] < 1.5 * data['atr']  # Если диапазон меньше 1.5 * ATR, считаем это боковиком

        # Определяем сегмент
        data['segment'] = np.where(data['is_flat'], 'flat', data['trend'])

        return data