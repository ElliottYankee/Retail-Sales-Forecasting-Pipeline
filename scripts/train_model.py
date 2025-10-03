import logging
import pandas as pd
import json

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
