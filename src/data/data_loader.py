import pandas as pd
import logging

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
    """Aggregate product-level data to store-category level."""
    agg_df = df.groupby(['Date', 'Store ID', 'Category', 'Region']).agg({
        'Units Sold': 'sum',
        'Inventory Level': 'sum',
        'Price': 'mean',
        'Discount': 'mean',
        'Holiday/Promotion': 'max',
        'Weather Condition': 'first',
        'Competitor Pricing': 'mean'
    }).reset_index()
    
    # Rename columns for consistency
    agg_df.columns = ['Date', 'Store_ID', 'Category', 'Region', 'Units_Sold',
                      'Inventory_Level', 'Avg_Price', 'Avg_Discount',
                      'Has_Promotion', 'Weather_Condition', 'Competitor_Price']
    
    logger.info(f"Aggregated to {len(agg_df):,} store-category records")
    
    return agg_df