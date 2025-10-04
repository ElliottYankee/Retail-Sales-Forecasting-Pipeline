import logging
import pandas as pd
import json
from pathlib import Path
import joblib
import sys

# Adding project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    # Saving trained models
    models_dir = Path('trained_models')
    models_dir.mkdir(exist_ok=True)  # Creating the models directory if it doesn't exist
    
    model_path = models_dir / 'xgboost_model.pkl'
    joblib.dump(xgb_model, model_path)
    logger.info(f"Model saved to: {model_path}")

    model_path = models_dir / 'randomforest_model.pkl'
    joblib.dump(rf_model, model_path)
    logger.info(f"Model saved to: {model_path}")

    # Saving feature columns with model for consistency
    features_path = models_dir / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    logger.info(f"Feature columns saved to: {features_path}")

    print("MODEL TRAINING COMPLETE")
    print("\nMODEL COMPARISON:")
    print(f"XGBoost MAPE: {xgb_metrics['MAPE']}%")
    print(f"RandomForest MAPE: {rf_metrics['MAPE']}%")

if __name__ == "__main__":
    main()
