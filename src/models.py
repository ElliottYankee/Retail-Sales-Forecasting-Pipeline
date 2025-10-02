import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
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
    
    def predict(self, X):
        """Generate predictions for feature matrix X"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Return feature importance scores"""
        return self.model.feature_importances_
    
class RandomForestForecaster:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=1,
            n_jobs=-1  # Using all CPU cores
        )
        logger.info("Initialized Random Forest model")