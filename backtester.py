import random
import numpy as np
import pandas as pd
from utils import log

class Backtester:
    def __init__(self, model, risk_manager):
        self.model = model
        self.risk_manager = risk_manager

    def run(self, data, iterations=100):
        """
        Проводит бэктестинг на исторических данных.
        """
        if data.empty:
            return 0.0, 0.0  # Возвращаем точность и прибыль/убыток

        try:
            correct_predictions = 0
            total_profit = 0.0

            for _ in range(iterations):
                # Случайным образом выбираем точку входа
                idx = random.randint(0, len(data) - 101)
                entry_point = data.iloc[idx]

                # Проверяем, что данные не содержат NaN
                if entry_point.isnull().any():
                    continue

                entry_price = entry_point["close"]
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
                        total_profit += (take_profit - entry_price) * self.risk_manager.calculate_position_size(1000, 0.01, entry_price, stop_loss)
                        break
                    elif current_price <= stop_loss:
                        total_profit += (stop_loss - entry_price) * self.risk_manager.calculate_position_size(1000, 0.01, entry_price, stop_loss)
                        break

            accuracy = correct_predictions / iterations
            profit_percentage = (total_profit / 1000) * 100  # Прибыль/убыток в процентах от депозита
            return accuracy, profit_percentage

        except Exception as e:
            log(f"Error during backtesting: {e}", level="error")
            return 0.0, 0.0