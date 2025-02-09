import random
import numpy as np
import pandas as pd
from utils import log

class Backtester:
    def __init__(self, model, risk_manager):
        self.model = model
        self.risk_manager = risk_manager

    def run(self, data, deposit, risk_per_trade, iterations=100):
        """
        Проводит бэктестинг на исторических данных.
        :param data: DataFrame с историческими данными.
        :param deposit: Начальный депозит.
        :param risk_per_trade: Риск на сделку (в процентах от депозита).
        :param iterations: Количество итераций для симуляции.
        :return: accuracy (точность), profit (прибыль/убыток в процентах от депозита).
        """
        if data.empty:
            return 0.0, 0.0  # Возвращаем точность и прибыль/убыток

        try:
            successful_trades = 0  # Счётчик успешных сделок
            total_profit = 0.0  # Общая прибыль/убыток
            initial_deposit = deposit  # Сохраняем начальный депозит

            for _ in range(iterations):
                idx = random.randint(0, len(data) - 101)  # Оставляем 100 свечей для анализа
                current_data = data.iloc[idx]

                required_columns = [
                    'momentum', 'volatility', 'rsi_14', 'ema_20', 'macd', 'macd_signal', 
                    'volume_profile', 'atr_14', 'upper_band', 'lower_band', 'adx_14', 'obv'
                ]
                X = current_data[required_columns].values.reshape(1, -1)
                signal = self.model.predict(X)[0]  # Предсказание модели (1 — лонг, 0 — шорт)

                entry_price = current_data["close"]
                atr = current_data["atr_14"]
                stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, atr)

                if signal == 0:
                    stop_loss, take_profit = take_profit, stop_loss

                position_size = self.risk_manager.calculate_position_size(deposit, risk_per_trade, entry_price, stop_loss)

                # Симулируем движение цены и закрытие позиции
                for i in range(idx + 1, idx + 101):
                    if i >= len(data):
                        break  # Если вышли за пределы данных, завершаем сделку

                    current_price = data.iloc[i]["close"]

                    if signal == 1:  # Лонг
                        if current_price >= take_profit:
                            successful_trades += 1
                            profit = (take_profit - entry_price) * position_size
                            total_profit += profit
                            deposit += profit
                            break
                        elif current_price <= stop_loss:
                            loss = (stop_loss - entry_price) * position_size
                            total_profit += loss
                            deposit += loss
                            break
                    elif signal == 0:  # Шорт
                        if current_price <= take_profit:
                            successful_trades += 1
                            profit = (entry_price - take_profit) * position_size
                            total_profit += profit
                            deposit += profit
                            break
                        elif current_price >= stop_loss:
                            loss = (entry_price - stop_loss) * position_size
                            total_profit += loss
                            deposit += loss
                            break

            accuracy = successful_trades / iterations
            profit_percentage = (total_profit / initial_deposit) * 100  # Прибыль/убыток в процентах от депозита

            log(f"Backtesting completed: Accuracy = {accuracy:.2%}, Profit/Loss = {profit_percentage:.2f}%")
            return accuracy, profit_percentage

        except Exception as e:
            log(f"Error during backtesting: {e}", level="error")
            return 0.0, 0.0