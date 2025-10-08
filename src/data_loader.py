import pandas as pd
from pathlib import Path
import logging
import sys

# Adding project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, RETAIL_STORE_INVENTORY

logger = logging.getLogger(__name__)

def assign_proper_seasons(month: int) -> str:
    """Assign seasons based on month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # [9, 10, 11]
        return 'Autumn'

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load raw data and apply basic cleaning"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Applying data quality fixes found in dataset overview notebook
    df = df.drop(['Demand Forecast'], axis=1, errors='ignore')
    df['Season'] = df['Date'].dt.month.map(assign_proper_seasons)

    logger.info(f"Loaded and cleaned {len(df):,} records")
    
    return df

def aggregate_to_store_category(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product-level data to store-category level"""
    agg_df = df.groupby(['Date', 'Store ID', 'Category', 'Region']).agg({
        'Inventory Level': 'sum',
        'Units Sold': 'sum',
        'Price': 'mean',
        'Discount': 'mean',
        'Weather Condition': 'first',
        'Holiday/Promotion': 'max',
        'Competitor Pricing': 'mean'
    }).reset_index()
    
    # Renaming columns for consistency
    agg_df.columns = ['Date', 'Store ID', 'Category', 'Region',
                      'Inventory Level', 'Units Sold', 'Average Price', 'Average Discount', 'Weather Condition',
                      'Holiday/Promotion', 'Competitor Pricing']
    
    # Rounding numeric columns to 2 decimal places for consistency
    numeric_cols = ['Average Price', 'Average Discount', 'Competitor Pricing']
    agg_df[numeric_cols] = agg_df[numeric_cols].round(2)
    
    logger.info(f"Aggregated to {len(agg_df):,} store-category records")

    return agg_df

if __name__ == "__main__":
    raw_file_path = RAW_DATA_DIR / RETAIL_STORE_INVENTORY
    
    # Running data cleansing and aggregation
    cleaned_df = load_and_clean_data(str(raw_file_path))
    aggregated_df = aggregate_to_store_category(cleaned_df)

    # Checking if aggregation worked correctly
    print(f"\nUnique stores: {aggregated_df['Store ID'].unique()}")
    print(f"Total records: {len(aggregated_df)}")
    print(f"Date range: {aggregated_df['Date'].min()} to {aggregated_df['Date'].max()}")

    # Checking store distribution
    print("\nRecords per store:")
    print(aggregated_df['Store ID'].value_counts().sort_index())
    
    # Displaying samples of cleaned and aggregated data
    print(f"\nCleaned Data Sample ({len(cleaned_df.columns)} columns):")
    print(cleaned_df.head(15))
    print(f"\nAggregated Data Sample ({len(aggregated_df.columns)} columns):")
    print(aggregated_df.head(15))