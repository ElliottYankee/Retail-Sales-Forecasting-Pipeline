import logging
import pandas as pd
import json

from src.models import XGBoostForecaster, RandomForestForecaster
from src.evaluation import evaluate_model, print_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("STAGE 2: MODEL TRAINING")

    # Loading processed data
    logger.info("Loading processed data...")
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')

    # Loading feature column names
    with open('data/processed/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    logger.info(f"Training with {len(feature_columns)} features on {len(train_data):,} records")

    # Preparing X (features) and y (target) for training
    X_train = train_data[feature_columns]
    y_train = train_data['Units Sold']
    X_test = test_data[feature_columns]
    y_test = test_data['Units Sold']

    # Training models
    logger.info("Training XGBoost model...")
    xgb_model = XGBoostForecaster()
    xgb_model.fit(X_train, y_train)

    logger.info("Training RandomForest model...")
    rf_model = RandomForestForecaster()
    rf_model.fit(X_train, y_train)

    # Evaluating models on test set
    logger.info("Evaluating XGBoost...")
    xgb_predictions = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_predictions)
    print_metrics(xgb_metrics)

    logger.info("Evaluating RandomForest...")
    rf_predictions = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_predictions)
    print_metrics(rf_metrics)
