import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import logging
import sys

# Adding project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, RANDOM_STATE, N_JOBS

logger = logging.getLogger(__name__)

class XGBoostForecaster:
    def __init__(self, xgb_params=XGBOOST_PARAMS):
        self.model = xgb.XGBRegressor(
            n_estimators= xgb_params['n_estimators'],
            max_depth= xgb_params['max_depth'],
            learning_rate= xgb_params['learning_rate'],
            random_state= RANDOM_STATE,
            n_jobs= N_JOBS
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
    def __init__(self, rf_params=RANDOM_FOREST_PARAMS):
        self.model = RandomForestRegressor(
            n_estimators= rf_params['n_estimators'],
            max_depth= rf_params['max_depth'],
            min_samples_split= rf_params['min_samples_split'],
            random_state= RANDOM_STATE,
            n_jobs= N_JOBS
        )
        logger.info("Initialized Random Forest model")

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