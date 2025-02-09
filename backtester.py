import random
import numpy as np
import pandas as pd
from utils import log

class Backtester:
    def __init__(self, models, lstm_models, risk_manager, required_columns):
        """
        Инициализация бэктестера.
        :param models: Словарь с моделями RandomForest для каждой торговой пары.
        :param lstm_models: Словарь с LSTM-моделями для каждой торговой пары.
        :param risk_manager: Менеджер рисков.
        :param required_columns: Список необходимых признаков для предсказания.
        """
        self.models = models
        self.lstm_models = lstm_models
        self.risk_manager = risk_manager
        self.required_columns = required_columns

    def run(self, all_data, deposit, risk_per_trade, iterations=100):
        """
        Проводит бэктестинг на исторических данных для всех торговых пар.
        :param all_data: Словарь с данными для каждой торговой пары.
        :param deposit: Начальный депозит.
        :param risk_per_trade: Риск на сделку (в процентах от депозита).
        :param iterations: Количество итераций для симуляции.
        :return: accuracy (точность), profit (прибыль/убыток в процентах от депозита).
        """
        if not all_data:
            return 0.0, 0.0  # Возвращаем точность и прибыль/убыток

        try:
            successful_trades = 0  # Счётчик успешных сделок
            total_profit = 0.0  # Общая прибыль/убыток
            initial_deposit = deposit  # Сохраняем начальный депозит

            for _ in range(iterations):
                # Выбираем случайную торговую пару
                symbol = random.choice(list(all_data.keys()))
                data = all_data[symbol]

                # Выбираем случайный момент в истории
                idx = random.randint(0, len(data) - 101)  # Оставляем 100 свечей для анализа
                current_data = data.iloc[idx]

                # Получаем сигнал от RandomForest
                X = current_data[self.required_columns].values.reshape(1, -1)
                direction, level = self.models[symbol].predict(X)
                signal_rf = 1 if direction[0] == 1 else -1  # Преобразуем направление в сигнал (1 — лонг, -1 — шорт)

                # Получаем сигнал от LSTM
                X_lstm, _, _ = prepare_lstm_data(current_data.to_frame().T)  # Преобразуем в DataFrame для LSTM
                predicted_price = self.lstm_models[symbol].predict(X_lstm)
                signal_lstm = 1 if predicted_price > current_data["close"] else -1

                # Объединяем сигналы (например, среднее значение)
                signal = (signal_rf + signal_lstm) / 2

                # Рассчитываем точки входа, стоп-лосса и тейк-профита
                entry_price = current_data["close"]
                atr = current_data["atr_14"]
                stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, atr)

                if signal == -1:
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