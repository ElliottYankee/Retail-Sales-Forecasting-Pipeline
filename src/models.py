import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class XGBoostForecaster:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=1
        )
        logger.info("Initialized XGBoost model")

    def fit(self, X, y):
        """Train the model on feature matrix X and target y"""
        self.model.fit(X, y)
        logger.info("Model training complete")
        return self