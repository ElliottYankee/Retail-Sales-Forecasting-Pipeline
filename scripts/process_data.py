from pathlib import Path
import logging

from src.data_loader import load_and_clean_data, aggregate_to_store_category

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("STAGE 1: DATA PROCESSING")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load and clean raw data
    logger.info("Loading raw data...")
    raw_path = project_root / 'data' / 'raw' / 'retail_store_inventory.csv'
    raw_data = load_and_clean_data(str(raw_path))

    # Aggregate to store-category level
    logger.info("Aggregating to store-category level...")
    agg_data = aggregate_to_store_category(raw_data)