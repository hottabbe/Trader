import logging

class RiskManager:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def calculate_risk_management(self, entry_price, atr, risk_reward_ratio=2):
        """
        Рассчитывает стоп-лосс и тейк-профит на основе ATR и соотношения риска к прибыли.
        """
        stop_loss = entry_price - (atr * risk_reward_ratio)
        take_profit = entry_price + (atr * risk_reward_ratio)
        return stop_loss, take_profit

    def calculate_position_size(self, deposit, risk_per_trade, entry_price, stop_loss):
        """
        Рассчитывает размер позиции на основе депозита, риска на сделку и ATR.
        """
        risk_amount = deposit * risk_per_trade
        position_size = risk_amount / abs(entry_price - stop_loss)
        return position_size