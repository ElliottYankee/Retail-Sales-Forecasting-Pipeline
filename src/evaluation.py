import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Calculate regression metrics comparing true vs predicted values"""
    
    # Mean Absolute Error = average prediction error in units
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error = penalizes large errors more
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R-squared = how much variance the model explains (0-1, higher is better)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error = error as % of actual sales
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'R2': round(r2, 3)
    }

def print_metrics(metrics):
    """Display metrics in readable format"""
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value}")