import pandas as pd

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
    
    return df