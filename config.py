from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'trained_models'
RETAIL_STORE_INVENTORY = 'retail_store_inventory.csv'
RETAIL_FEATURES_COMPLETE = 'retail_features_complete.csv'
TRAIN_DATA = 'train_data.csv'
TEST_DATA = 'test_data.csv'
FEATURE_COLUMNS = 'feature_columns.json'
TRAIN_TEST_SPLIT = 0.8
N_JOBS = -1  # Using all available CPU cores
RANDOM_STATE = 1

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