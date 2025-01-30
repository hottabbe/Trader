import xgboost as xgb
from sklearn.metrics import accuracy_score
import logging
import os

class TradingModel:
    def __init__(self, create_logs=True):
        self.model = xgb.XGBClassifier(random_state=42)
        self.create_logs = create_logs
        if create_logs:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def train(self, X_train, y_train):
        if X_train.empty or y_train.empty:
            if self.create_logs:
                self.logger.warning("Empty training data received. Skipping training.")
            return

        if self.create_logs:
            self.logger.info("Training the model")
        self.model.fit(X_train, y_train)
        if self.create_logs:
            self.logger.info("Model training completed")

    def predict(self, X):
        if X.empty:
            if self.create_logs:
                self.logger.warning("Empty DataFrame received for prediction. Returning default value.")
            return [0]  # Возвращаем значение по умолчанию

        if self.create_logs:
            self.logger.info("Making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if X_test.empty or y_test.empty:
            if self.create_logs:
                self.logger.warning("Empty test data received. Skipping evaluation.")
            return 0.0

        if self.create_logs:
            self.logger.info("Evaluating the model")
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        if self.create_logs:
            self.logger.info(f'Model accuracy on test data: {accuracy:.2%}')
        return accuracy

    def save_model(self, filename):
        if self.create_logs:
            self.logger.info(f"Saving model to {filename}")
        self.model.save_model(filename)

    def load_model(self, filename):
        if os.path.exists(filename):
            if self.create_logs:
                self.logger.info(f"Loading model from {filename}")
            self.model.load_model(filename)
        else:
            if self.create_logs:
                self.logger.warning(f"Model file {filename} not found.")