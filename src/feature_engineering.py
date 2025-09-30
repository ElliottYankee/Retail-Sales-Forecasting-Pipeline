import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features"""
    df = df.copy()
    
    # Basic time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Binary flags
    df['Is Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Is MonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding to capture circular nature of time
    df['Month Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling window features"""
    df = df.copy().sort_values(['Store ID', 'Category', 'Date'])
    
    for (store_id, category), group in df.groupby(['Store ID', 'Category']):
        mask = (df['Store ID'] == store_id) & (df['Category'] == category)
        
        # Lag features: gets sales for N days ago, useful for capturing time dependencies
        df.loc[mask, 'Sales Lag 7'] = group['Units Sold'].shift(7)
        df.loc[mask, 'Sales Lag 30'] = group['Units Sold'].shift(30)
        
        # Rolling averages: gets average over the past N days to smooth out noise
        df.loc[mask, 'Sales MA 7'] = group['Units Sold'].rolling(7).mean()
        df.loc[mask, 'Sales MA 30'] = group['Units Sold'].rolling(30).mean()
        
        # Rolling std: shows how volatile sales are, higher std = more unpredictable
        df.loc[mask, 'Sales Std 7'] = group['Units Sold'].rolling(7).std()
        
        # Growth rate: shows how much sales change week-over-week
        df.loc[mask, 'Sales Growth 7d'] = group['Units Sold'].pct_change(7)

    # Filling NaN with 0 to represent no history
    lag_cols = [col for col in df.columns if any(x in col for x in ['Lag ', 'MA ', 'Std ', 'Growth '])]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    return df

def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-relevant derived features"""
    df = df.copy()

    # Price competitiveness: Ratio > 1 = competitor is less expensive
    df['Price vs Competitor'] = df['Avg Price'] / (df['Competitor Price'] + 0.01)
    
    # Inventory management: High value = overstock risk
    df['Inventory Days Supply'] = df['Inventory Level'] / (df['Units Sold'] + 1)

    # Weather impact score
    weather_scores = {'Sunny': 1.05, 'Cloudy': 1.0, 'Rainy': 0.95, 'Snowy': 0.90}
    df['Weather Impact'] = df['Weather Condition'].map(weather_scores)
    
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical variables"""
    df = df.copy()
    
    # Encoding categoricals
    for col in ['Category', 'Region', 'Weather Condition']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
    
    return df

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence"""
    df = df.copy()
    
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_business_features(df)
    df = encode_categoricals(df)
    
    return df