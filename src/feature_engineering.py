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
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Is_MonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding to capture circular nature of time
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    return df

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling window features"""
    
    for (store_id, category), group in df.groupby(['Store ID', 'Category']):
        mask = (df['Store ID'] == store_id) & (df['Category'] == category)
        
        # Lag features: gets sales for N days ago, useful for capturing time dependencies
        df.loc[mask, 'Sales Lag 7'] = group['Units Sold'].shift(7)
        
        # Rolling averages: gets average over the past N days to smooth out noise
        df.loc[mask, 'Sales MA 7'] = group['Units Sold'].rolling(7).mean()
        
        # Rolling std: shows how volatile sales are, higher std = more unpredictable
        df.loc[mask, 'Sales Std 7'] = group['Units Sold'].rolling(7).std()
        
        # Growth rate: shows how much sales change week-over-week
        df.loc[mask, 'Sales Growth 7d'] = group['Units Sold'].pct_change(7)

    # Fill NaN with 0 to represent no history
    lag_cols = [col for col in df.columns if any(x in col for x in ['Lag_', 'MA_', 'Std_', 'Growth_'])]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    return df

