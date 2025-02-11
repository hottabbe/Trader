import random
import numpy as np
import pandas as pd
from utils import log, prepare_lstm_data

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

                # Убедимся, что данных достаточно для анализа
                if len(data) < 500:
                    log(f"Not enough data for {symbol}. Skipping.", level="warning")
                    continue

                # Выбираем случайный момент в истории (оставляем 500 свечей для анализа)
                idx = random.randint(500, len(data) - 1)
                current_data = data.iloc[idx - 500:idx]  # Берём данные за последние 500 свечей

                # Проверяем, что все необходимые колонки присутствуют
                if not all(col in current_data.columns for col in self.required_columns):
                    log(f"Missing required columns for {symbol}. Skipping.", level="warning")
                    continue

                # Получаем сигнал от RandomForest
                X = current_data[self.required_columns].values[-1].reshape(1, -1)  # Берём последнюю строку
                direction, level = self.models[symbol].predict(X)
                signal_rf = 1 if direction[0] == 1 else -1  # Преобразуем направление в сигнал (1 — лонг, -1 — шорт)

                # Получаем сигнал от LSTM
                X_lstm, _, _ = prepare_lstm_data(current_data)
                if X_lstm.size == 0:
                    log(f"Not enough data for LSTM prediction for {symbol}. Skipping.", level="warning")
                    continue

                predicted_price = self.lstm_models[symbol].predict(X_lstm[-1].reshape(1, *X_lstm.shape[1:]))
                signal_lstm = 1 if predicted_price > current_data["close"].iloc[-1] else -1

                # Объединяем сигналы (например, среднее значение)
                signal = (signal_rf + signal_lstm) / 2

                # Рассчитываем точки входа, стоп-лосса и тейк-профита
                entry_price = current_data["close"].iloc[-1]
                atr = current_data["atr_14"].iloc[-1]

                # Проверяем, что ATR и цена входа корректны
                if atr <= 0 or entry_price <= 0:
                    log(f"Invalid ATR or entry price for {symbol}. Skipping.", level="warning")
                    continue

                stop_loss, take_profit = self.risk_manager.calculate_risk_management(entry_price, atr)

                if signal == -1:
                    stop_loss, take_profit = take_profit, stop_loss

                # Рассчитываем размер позиции
                position_size = self.risk_manager.calculate_position_size(deposit, risk_per_trade, entry_price, stop_loss)

                # Симулируем движение цены и закрытие позиции
                for i in range(idx, len(data)):
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
                    elif signal == -1:  # Шорт
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