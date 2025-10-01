from pathlib import Path
import logging

from src.data_loader import load_and_clean_data, aggregate_to_store_category
from src.feature_engineering import create_all_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("STAGE 1: DATA PROCESSING")
    
    # Getting project root
    project_root = Path(__file__).parent.parent
    
    # Loading and cleaning raw data
    logger.info("Loading raw data...")
    raw_path = project_root / 'data' / 'raw' / 'retail_store_inventory.csv'
    raw_data = load_and_clean_data(str(raw_path))

    # Aggregating to store-category level
    logger.info("Aggregating to store-category level...")
    agg_data = aggregate_to_store_category(raw_data)

    # Creating all features
    logger.info("Engineering features...")
    features_data = create_all_features(agg_data)

    # Creating processed directory
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Saving complete processed dataset
    complete_path = processed_dir / 'retail_features_complete.csv'
    features_data.to_csv(complete_path, index=False)
    logger.info(f"Saved complete dataset: {complete_path}")