from pathlib import Path
import logging
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    # Creating time-based train/test splits (80/20)
    logger.info("Creating train/test splits...")
    features_data = features_data.sort_values('Date')
    cutoff_idx = int(len(features_data) * 0.8)
    
    train_data = features_data.iloc[:cutoff_idx]
    test_data = features_data.iloc[cutoff_idx:]

    # Saving splits
    train_path = processed_dir / 'train_data.csv'
    test_path = processed_dir / 'test_data.csv'
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(f"Train set: {len(train_data):,} records ({train_data['Date'].min().date()} to {train_data['Date'].max().date()})")
    logger.info(f"Test set: {len(test_data):,} records ({test_data['Date'].min().date()} to {test_data['Date'].max().date()})")

    # Saving feature column names (exclude identifiers and target)
    feature_cols = [col for col in features_data.columns 
                   if col not in ['Date', 'Store ID', 'Category', 'Region', 'Units Sold']]
    
    features_path = processed_dir / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    logger.info(f"Saved {len(feature_cols)} feature columns")

    print("DATA PROCESSING COMPLETE")
    print(f"Processed data saved to: {processed_dir}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Train records: {len(train_data):,}")
    print(f"Test records: {len(test_data):,}")

if __name__ == "__main__":
    main()

    