from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'trained_models'

# Model hyperparams
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5
}