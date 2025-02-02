import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import log

class TradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False  # Флаг, указывающий, обучена ли модель

    def train(self, X_train, y_train):
        """
        Обучает модель на предоставленных данных.
        """
        try:
            if X_train.size == 0 or y_train.size == 0:
                log("Training data is empty. Skipping training.", level="warning")
                return

            if np.isnan(X_train).any() or np.isnan(y_train).any():
                log("Training data contains NaN values. Cleaning data...", level="warning")
                mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                X_train = X_train[mask]
                y_train = y_train[mask]

            log("Training the model")
            self.model.fit(X_train, y_train)
            self.is_trained = True
            log("Model training completed successfully")
        except Exception as e:
            log(f"Error during model training: {e}", level="error")
            raise

    def predict(self, X):
        """
        Предсказывает сигнал на основе предоставленных данных.
        """
        try:
            if X.size == 0:
                log("Prediction data is empty. Skipping prediction.", level="warning")
                return np.array([])

            if not self.is_trained:
                raise ValueError("Model is not trained. Call 'train' before making predictions.")

            log("Predicting signals")
            return self.model.predict(X)
        except Exception as e:
            log(f"Error during prediction: {e}", level="error")
            raise

    def evaluate(self, X_test, y_test):
        """
        Оценивает точность модели на тестовых данных.
        """
        try:
            if X_test.size == 0 or y_test.size == 0:
                log("Test data is empty. Skipping evaluation.", level="warning")
                return 0.0

            if np.isnan(X_test).any() or np.isnan(y_test).any():
                log("Test data contains NaN values. Cleaning data...", level="warning")
                mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
                X_test = X_test[mask]
                y_test = y_test[mask]

            log("Evaluating the model")
            accuracy = self.model.score(X_test, y_test)
            log(f"Model evaluation completed with accuracy: {accuracy:.2%}")
            return accuracy
        except Exception as e:
            log(f"Error during model evaluation: {e}", level="error")
            raise