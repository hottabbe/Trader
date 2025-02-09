class RiskManager:
    def calculate_risk_management(self, entry_price, atr, risk_reward_ratio=2):
        """
        Рассчитывает стоп-лосс и тейк-профит на основе ATR и соотношения риска к прибыли.
        :param entry_price: Цена входа.
        :param atr: Average True Range (ATR).
        :param risk_reward_ratio: Соотношение риска к прибыли (по умолчанию 1:2).
        :return: stop_loss, take_profit
        """
        stop_loss = entry_price - atr
        take_profit = entry_price + atr * risk_reward_ratio
        return stop_loss, take_profit

    def calculate_position_size(self, deposit, risk_per_trade, entry_price, stop_loss):
        """
        Рассчитывает размер позиции на основе депозита, риска на сделку и ATR.
        :param deposit: Размер депозита.
        :param risk_per_trade: Риск на сделку (в процентах от депозита).
        :param entry_price: Цена входа.
        :param stop_loss: Цена стоп-лосса.
        :return: Размер позиции.
        """
        risk_amount = deposit * risk_per_trade
        position_size = risk_amount / abs(entry_price - stop_loss)
        return position_size