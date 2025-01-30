import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, model):
        self.model = model

    def run(self, data):
        """
        Запускает бэктест на исторических данных.
        """
        if data.empty:
            return 0.0

        # Выбираем необходимые колонки
        required_columns = ['momentum', 'volatility', 'rsi', 'ema_20', 'macd', 'macd_signal', 'volume_profile', 'atr', 'upper_band', 'lower_band', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'adx', 'obv']
        X = data[required_columns].values
        y = data["signal"].values

        # Прогнозируем сигналы
        predictions = self.model.predict(X)

        # Рассчитываем доходность
        returns = data["return"].values
        strategy_returns = returns * predictions

        # Суммарная доходность
        total_return = np.prod(1 + strategy_returns) - 1

        return total_return
