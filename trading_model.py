import numpy as np
from utils import log
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class TradingModel:
    def __init__(self):
        # Модель для классификации (направление)
        self.classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=42,verbose=2,n_jobs=-1)
        
        # Модель для регрессии (уровень)
        self.regressor = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=42,verbose=2,n_jobs=-1)
        
        self.is_trained = False  # Флаг, указывающий, обучена ли модель

    def train(self, X_train, y_train_direction, y_train_level):
        """
        Обучает модель на предоставленных данных.
        :param X_train: Признаки для обучения.
        :param y_train_direction: Целевая переменная для классификации (направление).
        :param y_train_level: Целевая переменная для регрессии (уровень).
        """
        try:
            if X_train.size == 0 or y_train_direction.size == 0 or y_train_level.size == 0:
                log("Training data is empty. Skipping training.", level="warning")
                return

            if np.isnan(X_train).any() or np.isnan(y_train_direction).any() or np.isnan(y_train_level).any():
                log("Training data contains NaN values. Cleaning data...", level="warning")
                mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train_direction) & ~np.isnan(y_train_level)
                X_train = X_train[mask]
                y_train_direction = y_train_direction[mask]
                y_train_level = y_train_level[mask]

            log("Training the classifier (direction)")
            self.classifier.fit(X_train, y_train_direction)

            log("Training the regressor (level)")
            self.regressor.fit(X_train, y_train_level)

            self.is_trained = True
            log("Model training completed successfully")

            # Оценка модели с помощью кросс-валидации
            scores_direction = cross_val_score(self.classifier, X_train, y_train_direction, cv=5, scoring='accuracy')
            log(f"Cross-validation accuracy (direction): {scores_direction.mean():.2%}")

            scores_level = cross_val_score(self.regressor, X_train, y_train_level, cv=5, scoring='neg_mean_squared_error')
            log(f"Cross-validation MSE (level): {-scores_level.mean():.4f}")

        except Exception as e:
            log(f"Error during model training: {e}", level="error")
            raise

    def predict(self, X):
        """
        Предсказывает направление и уровень изменения цены.
        :param X: Входные данные.
        :return: Прогнозируемые значения (направление и уровень).
        """
        try:
            if X.size == 0:
                log("Prediction data is empty. Skipping prediction.", level="warning")
                return np.array([]), np.array([])

            if not self.is_trained:
                raise ValueError("Model is not trained. Call 'train' before making predictions.")

            log("Predicting signals and levels")
            direction = self.classifier.predict(X)
            level = self.regressor.predict(X)
            return direction, level
        except Exception as e:
            log(f"Error during prediction: {e}", level="error")
            raise

    def evaluate(self, X_test, y_test_direction, y_test_level):
        """
        Оценивает точность модели на тестовых данных.
        :param X_test: Признаки для тестирования.
        :param y_test_direction: Целевая переменная для классификации (направление).
        :param y_test_level: Целевая переменная для регрессии (уровень).
        :return: Точность классификации и MSE для регрессии.
        """
        try:
            if X_test.size == 0 or y_test_direction.size == 0 or y_test_level.size == 0:
                log("Test data is empty. Skipping evaluation.", level="warning")
                return 0.0, 0.0

            if np.isnan(X_test).any() or np.isnan(y_test_direction).any() or np.isnan(y_test_level).any():
                log("Test data contains NaN values. Cleaning data...", level="warning")
                mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test_direction) & ~np.isnan(y_test_level)
                X_test = X_test[mask]
                y_test_direction = y_test_direction[mask]
                y_test_level = y_test_level[mask]

            log("Evaluating the classifier (direction)")
            accuracy = self.classifier.score(X_test, y_test_direction)
            log(f"Classifier evaluation completed with accuracy: {accuracy:.2%}")

            log("Evaluating the regressor (level)")
            mse = np.mean((self.regressor.predict(X_test) - y_test_level) ** 2)
            log(f"Regressor evaluation completed with MSE: {mse:.4f}")

            return accuracy, mse
        except Exception as e:
            log(f"Error during model evaluation: {e}", level="error")
            raise