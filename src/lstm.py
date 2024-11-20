import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import glob
import logging
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from data.modify_dataset import prepare_data

# Ensure the logs, models, and params directories exist
LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Generate a timestamped log file name
log_filename = os.path.join(LOG_DIR, f"lstm_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging with both console and file handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers if any
if logger.hasHandlers():
    logger.handlers.clear()

# Create a console handler for real-time terminal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Create a file handler for saving logs
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Function to keep only the most recent 5 logs
def cleanup_old_logs(directory, prefix="lstm_log_", max_logs=5):
    log_files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.log")), key=os.path.getmtime)
    if len(log_files) > max_logs:
        for log_file in log_files[:-max_logs]:
            os.remove(log_file)
            logger.info(f"Deleted old log file: {log_file}")

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

@tf.function(reduce_retracing=True)
def make_prediction(model, X_test):
    return model(X_test, training=False)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, SEQ_LENGTH, scaler):
    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH, 1)))  # Define the input shape explicitly
    model.add(LSTM(units=int(num_units), activation='relu'))  # Ensure units is an integer
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model and capture the training history
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    # Log training and validation loss
    final_training_loss = history.history['loss'][-1]  # Get the last value of training loss
    final_val_loss = history.history['val_loss'][-1]   # Get the last value of validation loss

    # Evaluate the model
    predictions = make_prediction(model, tf.convert_to_tensor(X_test, dtype=tf.float32))
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    # Ensure the return matches the number of values expected to be unpacked
    return model, mae, rmse, final_training_loss, final_val_loss 

def plot_metrics(metrics_history, param_name):
    param_values = [entry[param_name] for entry in metrics_history]
    mae_values = [entry['mae'] for entry in metrics_history]
    rmse_values = [entry['rmse'] for entry in metrics_history]
    training_loss_values = [entry['training_loss'] for entry in metrics_history]
    val_loss_values = [entry['val_loss'] for entry in metrics_history]

    plt.figure(figsize=(14, 7))
    plt.plot(param_values, mae_values, marker='o', label='MAE')
    plt.plot(param_values, rmse_values, marker='o', label='RMSE')
    plt.plot(param_values, training_loss_values, marker='o', label='Training Loss')
    plt.plot(param_values, val_loss_values, marker='o', label='Validation Loss')
    plt.xlabel(param_name.capitalize())
    plt.ylabel('Metric Value')
    plt.title(f'Metrics vs {param_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_best_params(best_params):
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')
    
    # Convert all values in best_params to native Python types
    best_params_converted = {key: int(value) if isinstance(value, (np.integer, np.int64)) else value
                             for key, value in best_params.items()}
    
    # Check if the file exists and load existing parameters
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            existing_params = json.load(f)
        if isinstance(existing_params, dict):
            existing_params = [existing_params]  # Ensure it's a list
    else:
        existing_params = []

    # Append the new best parameters to the list
    existing_params.append(best_params_converted)

    # Convert any potential NumPy types in existing_params to native Python types
    existing_params = [{key: int(value) if isinstance(value, (np.integer, np.int64)) else value
                        for key, value in param.items()} for param in existing_params]

    # Save the updated parameter history
    with open(params_path, 'w') as f:
        json.dump(existing_params, f, indent=4)
    logger.info(f"Best parameters updated and saved to {params_path}")

def load_best_params():
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params_list = json.load(f)
            if isinstance(params_list, list) and params_list:
                return params_list[-1] 
    return None

def clear_params():
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')
    if os.path.exists(params_path):
        os.remove(params_path)
        logger.info(f"Cleared parameter file: {params_path}")

def remove_existing_model(model_dir):
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path) and file_path.endswith('.keras'):
            os.remove(file_path)
            logger.info(f"Deleted old model file: {file_path}")
            
def test_parameters(param_name, param_list, best_params, X_train, X_test, y_train, y_test, SEQ_LENGTH, scaler, test_count=20):
    metrics_history = []
    tested_count = 0
    unique_param_values = set(param_list)
    explored_combinations = set()

    def is_combination_explored(params):
        param_tuple = tuple(sorted(params.items()))
        return param_tuple in explored_combinations

    def add_combination_to_explored(params):
        param_tuple = tuple(sorted(params.items()))
        explored_combinations.add(param_tuple)

    while tested_count < test_count:
        # Ensure there are enough unique values to test
        while len(unique_param_values) < test_count:
            increment = np.random.randint(-50, 50) if param_name == 'num_units' else np.random.randint(-16, 16)
            new_value = max(10, best_params[param_name] + increment) if param_name == 'num_units' else max(1, best_params[param_name] + increment)
            if new_value not in unique_param_values:
                unique_param_values.add(new_value)

        # Iterate over unique parameter values and test them
        for param_value in list(unique_param_values):
            if tested_count >= test_count:
                break

            temp_params = best_params.copy()
            temp_params[param_name] = param_value

            if is_combination_explored(temp_params):
                logger.info(f"Skipping already tested combination: {temp_params}")
                unique_param_values.remove(param_value)  # Remove to avoid re-checking
                continue

            add_combination_to_explored(temp_params)

            logger.info(f"Testing {param_name}={param_value}")
            model, mae, rmse, training_loss, val_loss = train_and_evaluate_model(
                X_train, X_test, y_train, y_test, 
                temp_params['num_units'], temp_params['batch_size'], 
                temp_params['epochs'], 0.001, SEQ_LENGTH, scaler
            )
            metrics_history.append({
                param_name: param_value, 
                'mae': mae, 
                'rmse': rmse, 
                'training_loss': training_loss, 
                'val_loss': val_loss
            })
            logger.info(f"{param_name}={param_value} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Training Loss: {training_loss:.4f}, Final Val Loss: {val_loss:.4f}")

            tested_count += 1
            unique_param_values.remove(param_value)  # Ensure the tested value is not reused

            if tested_count >= test_count:
                break

    return metrics_history

def summarize_best_results(best_metrics_history):
    if not best_metrics_history:
        logger.warning("No best metrics found to summarize.")
        return None

    avg_mae = np.mean([entry['mae'] for entry in best_metrics_history])
    avg_rmse = np.mean([entry['rmse'] for entry in best_metrics_history])
    avg_training_loss = np.mean([entry['training_loss'] for entry in best_metrics_history])
    avg_val_loss = np.mean([entry['val_loss'] for entry in best_metrics_history])
    
    logger.info(f"Summary of best parameters:")
    logger.info(f"Average MAE: {avg_mae:.2f}")
    logger.info(f"Average RMSE: {avg_rmse:.2f}")
    logger.info(f"Average Training Loss: {avg_training_loss:.4f}")
    logger.info(f"Average Final Validation Loss: {avg_val_loss:.4f}")

    return avg_mae, avg_rmse, avg_training_loss, avg_val_loss


def main():
    cleanup_old_logs(LOG_DIR)

    # Option to clear saved parameters
    # if len(sys.argv) > 1 and sys.argv[1] == 'clear_params':
    #     clear_params()
    #     return

    logger.info("Starting LSTM model script")

    # Load and prepare data
    logger.info("Loading and preparing data")
    data = prepare_data('data/cafecast_data.xlsx')
    daily_data = data.resample('D')['transaction_qty'].sum()
    logger.info(f"Data overview: {daily_data.describe()}")

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))

    SEQ_LENGTH = 10  # Base sequence length
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

    # Load existing best parameters if available
    best_params = load_best_params() or {'num_units': 100, 'batch_size': 32, 'epochs': 50}
    logger.info(f"Starting with initial best parameters: {best_params}")
    
    best_metrics_history = []

    # Step 1: Find the best num_units by adjusting values around the current best
    num_units_list = [best_params['num_units']] + [
        max(10, best_params['num_units'] + delta) for delta in np.linspace(-50, 50, 5, dtype=int) if delta != 0
    ]
    metrics_history = test_parameters('num_units', num_units_list, best_params, X_train, X_test, y_train, y_test, SEQ_LENGTH, scaler)

    # Plot and find the best num_units
    plot_metrics(metrics_history, 'num_units')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['num_units'] = best_entry['num_units']
    logger.info(f"Best num_units found: {best_params['num_units']} with RMSE: {best_entry['rmse']:.2f}")

    # Step 2: Find the best batch_size by adjusting values around the current best
    batch_size_list = [best_params['batch_size']] + [
        max(1, best_params['batch_size'] + delta) for delta in np.linspace(-16, 16, 5, dtype=int) if delta != 0
    ]
    metrics_history = test_parameters('batch_size', batch_size_list, best_params, X_train, X_test, y_train, y_test, SEQ_LENGTH, scaler)

    # Plot and find the best batch_size
    plot_metrics(metrics_history, 'batch_size')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['batch_size'] = best_entry['batch_size']
    logger.info(f"Best batch_size found: {best_params['batch_size']} with RMSE: {best_entry['rmse']:.2f}")

    # Step 3: Find the best epochs by adjusting values around the current best
    epochs_list = [best_params['epochs']] + [
        max(10, best_params['epochs'] + delta) for delta in np.linspace(-50, 50, 5, dtype=int) if delta != 0
    ]
    metrics_history = test_parameters('epochs', epochs_list, best_params, X_train, X_test, y_train, y_test, SEQ_LENGTH, scaler)

    # Plot and find the best epochs
    plot_metrics(metrics_history, 'epochs')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['epochs'] = best_entry['epochs']
    logger.info(f"Best epochs found: {best_params['epochs']} with RMSE: {best_entry['rmse']:.2f}")

    # Save the updated best parameters
    save_best_params(best_params)
    
    summary = summarize_best_results(best_metrics_history)
    if summary:
        avg_mae, avg_rmse, avg_training_loss, avg_val_loss = summary
    else:
        logger.error("Failed to generate a summary of the best results.")
        # Remove existing model before saving the new best model
        remove_existing_model(MODEL_DIR)

    # Save the final best model
    final_model, mae, rmse, training_loss, val_loss = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        best_params['num_units'], best_params['batch_size'],
        best_params['epochs'], 0.001, SEQ_LENGTH, scaler
    )
    model_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    final_model.save(model_path)
    logger.info(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
