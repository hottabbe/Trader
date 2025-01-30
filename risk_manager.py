import logging

class RiskManager:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_risk_management(entry_price, atr, risk_reward_ratio=2):
        stop_loss = entry_price - (atr * risk_reward_ratio)
        take_profit = entry_price + (atr * risk_reward_ratio)
        return stop_loss, take_profit

    @staticmethod
    def calculate_position_size(deposit, risk_per_trade, entry_price, stop_loss):
        risk_amount = deposit * risk_per_trade
        position_size = risk_amount / abs(entry_price - stop_loss)
        return position_size