import os
import json
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.modify_dataset import prepare_data

# Ensure the logs, models, and params directories exist
LOG_DIR = 'logs/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Generate a timestamped log file name
log_filename = os.path.join(LOG_DIR, f"arima_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers if any
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler for real-time terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler for saving logs
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Function to cleanup old log files
def cleanup_old_logs(directory, prefix="arima_log_", max_logs=5):
    log_files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.log")), key=os.path.getmtime)
    if len(log_files) > max_logs:
        for log_file in log_files[:-max_logs]:
            os.remove(log_file)
            logger.info(f"Deleted old log file: {log_file}")

# Function to save the best ARIMA parameters
def save_best_params(params):
    params_path = os.path.join(PARAMS_DIR, 'best_arima_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            existing_params = json.load(f)
    else:
        existing_params = []

    # Append the new parameters
    existing_params.append(params)

    with open(params_path, 'w') as f:
        json.dump(existing_params, f, indent=4)
    logger.info(f"Best ARIMA parameters saved to {params_path}")

# Function to load the best ARIMA parameters
def load_best_params():
    params_path = os.path.join(PARAMS_DIR, 'best_arima_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params_list = json.load(f)
            return params_list[-1] if params_list else None
    return None

def clear_params():
    params_path = os.path.join(PARAMS_DIR, 'best_arima_params.json')
    if os.path.exists(params_path):
        os.remove(params_path)
        logger.info(f"Deleted parameter file: {params_path}")

# Function to evaluate the ARIMA model
def evaluate_arima_model(train, test, p, d, q):
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        final_val_loss = model_fit.bic

        return mae, rmse, final_val_loss
    except Exception as e:
        logger.warning(f"Failed to fit ARIMA({p},{d},{q}): {e}")
        return float('inf'), float('inf'), float('inf')

# Function to test multiple ARIMA parameters
def test_arima_parameters(train, test, best_params, param_name, param_range):
    metrics_history = []
    for param_value in param_range:
        temp_params = best_params.copy()
        temp_params[param_name] = param_value

        logger.info(f"Testing ARIMA parameters: {temp_params}")
        mae, rmse, val_loss = evaluate_arima_model(
            train, test, temp_params['p'], temp_params['d'], temp_params['q']
        )

        metrics_history.append({
            param_name: param_value,
            'mae': mae,
            'rmse': rmse,
            'val_loss': val_loss
        })
        logger.info(f"ARIMA{temp_params} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Final Val Loss (BIC): {val_loss:.2f}")

    return metrics_history

# Function to plot metrics
def plot_metrics(metrics_history, param_name):
    param_values = [entry[param_name] for entry in metrics_history]
    mae_values = [entry['mae'] for entry in metrics_history]
    rmse_values = [entry['rmse'] for entry in metrics_history]
    val_loss_values = [entry['val_loss'] for entry in metrics_history]

    plt.figure(figsize=(14, 7))
    plt.plot(param_values, mae_values, marker='o', label='MAE')
    plt.plot(param_values, rmse_values, marker='o', label='RMSE')
    plt.plot(param_values, val_loss_values, marker='o', label='Validation Loss (BIC)')
    plt.xlabel(param_name.capitalize())
    plt.ylabel('Metric Value')
    plt.title(f'Metrics vs {param_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    cleanup_old_logs(LOG_DIR)
    logger.info("Starting ARIMA model script")
    
    # Optionally clear parameters and remove old models
    # if 'clear_params' in sys.argv:
    #     clear_params()
    #     return

    # Load and prepare data
    logger.info("Loading and preparing data")
    data = prepare_data('data/cafecast_data.xlsx')
    daily_data = data.resample('D')['transaction_qty'].sum()

    # Train-test split
    train_size = int(len(daily_data) * 0.8)
    train, test = daily_data[:train_size], daily_data[train_size:]

    # Load the best parameters if available
    best_params = load_best_params() or {'p': 1, 'd': 1, 'q': 1}
    logger.info(f"Starting with initial best parameters: {best_params}")

    # Test ARIMA parameters for p
    p_range = range(max(0, best_params['p'] - 2), best_params['p'] + 30)
    metrics_history = test_arima_parameters(train, test, best_params, 'p', p_range)
    plot_metrics(metrics_history, 'p')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['p'] = best_entry['p']

    # Test ARIMA parameters for d
    d_range = range(max(0, best_params['d'] - 1), best_params['d'] + 20)
    metrics_history = test_arima_parameters(train, test, best_params, 'd', d_range)
    plot_metrics(metrics_history, 'd')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['d'] = best_entry['d']

    # Test ARIMA parameters for q
    q_range = range(max(0, best_params['q'] - 2), best_params['q'] + 30)
    metrics_history = test_arima_parameters(train, test, best_params, 'q', q_range)
    plot_metrics(metrics_history, 'q')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['q'] = best_entry['q']

    # Save the updated best parameters
    save_best_params(best_params)

    # Final evaluation with the best parameters
    logger.info(f"Evaluating final ARIMA model with parameters: {best_params}")
    mae, rmse, _ = evaluate_arima_model(train, test, best_params['p'], best_params['d'], best_params['q'])
    logger.info(f"Final ARIMA Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
