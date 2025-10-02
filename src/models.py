import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class XGBoostForecaster:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        logger.info("Initialized XGBoost model")