import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
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