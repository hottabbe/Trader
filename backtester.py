import random
import numpy as np
import pandas as pd
from utils import log

class Backtester:
    def __init__(self, model, risk_manager):
        """
        Инициализация бэктестера.
        :param model: Объект модели TradingModel.
        :param risk_manager: Объект RiskManager для управления рисками.
        """
        self.model = model
        self.risk_manager = risk_manager  # Добавляем RiskManager

    def run(self, data, iterations=100):
        """
        Проводит бэктестинг на исторических данных.
        :param data: DataFrame с историческими данными.
        :param iterations: Количество итераций для бэктестинга.
        :return: Точность модели в процентах.
        """
        if data.empty:
            log("No data available for backtesting. Skipping.", level="warning")
            return 0.0

        try:
            log(f"Running backtest with {iterations} iterations")
            correct_predictions = 0

            for _ in range(iterations):
                # Случайным образом выбираем точку входа
                idx = random.randint(0, len(data) - 101)
                entry_point = data.iloc[idx]
                entry_price = entry_point["close"]

                # Получаем ATR для расчета рисков
                atr = entry_point["atr"]

                # Рассчитываем стоп-лосс и тейк-профит
                stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, atr)

                # Проверяем, достигнут ли тейк-профит или стоп-лосс в следующие 100 свечей
                for i in range(idx + 1, idx + 101):
                    if i >= len(data):
                        break

                    current_price = data.iloc[i]["close"]
                    if current_price >= take_profit:
                        correct_predictions += 1
                        break
                    elif current_price <= stop_loss:
                        break

            accuracy = correct_predictions / iterations
            log(f"Backtest completed with accuracy: {accuracy:.2%}")
            return accuracy

        except Exception as e:
            log(f"Error during backtesting: {e}", level="error")
            return 0.0
