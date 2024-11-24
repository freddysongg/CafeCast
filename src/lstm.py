import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import logging
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from preprocess_data import process_data

LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"lstm_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(log_filename, mode='w')])

logger = logging.getLogger()

def create_sequences(data, seq_length, target_indices):
    """
    Creates input-output sequences for time-series forecasting.

    Args:
        data (np.array): Array of shape (time_steps, features).
        seq_length (int): Length of each input sequence.
        target_indices (list): Indices of target columns in the data.

    Returns:
        np.array, np.array: Input sequences (X) and target outputs (y).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_indices])
    return np.array(X), np.array(y)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, seq_length, target_size):
    """
    Trains and evaluates an LSTM model.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        num_units (int): Number of units in the LSTM layer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        seq_length (int): Sequence length for the input data.
        target_size (int): Number of output targets.

    Returns:
        Trained model, float, float: Trained model, MAE, and RMSE.
    """
    model = Sequential([
        Input(shape=(seq_length, X_train.shape[2])),
        LSTM(units=num_units, activation='relu'),
        Dense(target_size)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return model, mae, rmse, history.history['val_loss'][-1]

def save_best_params(best_params):
    """
    Saves the best parameters dynamically to a JSON file. Handles both single
    parameter sets and lists of parameter sets.

    Args:
        best_params (dict): A dictionary containing the best parameters to save.
    """
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')

    required_keys = ['num_units', 'batch_size', 'epochs', 'learning_rate']
    for key in required_keys:
        if key not in best_params:
            logger.warning(f"Missing parameter '{key}' in best_params. Adding default value.")
            if key == 'learning_rate':
                best_params[key] = 0.001
            elif key == 'num_units':
                best_params[key] = 128
            elif key == 'batch_size':
                best_params[key] = 32
            elif key == 'epochs':
                best_params[key] = 50

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            try:
                existing_params = json.load(f)
                if not isinstance(existing_params, list):
                    logger.warning(f"Existing params are not in list format. Overwriting with a list.")
                    existing_params = []
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON. Overwriting with a new list.")
                existing_params = []
    else:
        existing_params = []

    existing_params.append(best_params)

    with open(params_path, 'w') as f:
        json.dump(existing_params, f, indent=4)
    logger.info(f"Best parameters appended and saved to {params_path}")


def load_best_params():
    """
    Loads the most recent parameters from a JSON file. If the file contains a list,
    use the last entry. If keys are missing, default values are added.

    Returns:
        dict: Best parameters with all required keys.
    """
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')

    default_params = {'num_units': 128, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            try:
                params = json.load(f)

                if isinstance(params, list) and len(params) > 0:
                    params = params[-1]  # Use the last entry
                elif not isinstance(params, dict):
                    logger.warning(f"Invalid format in {params_path}. Using default parameters.")
                    return default_params

                for key, default_value in default_params.items():
                    if key not in params:
                        logger.warning(f"Key '{key}' missing in loaded params. Adding default value: {default_value}")
                        params[key] = default_value

                return params
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON in {params_path}. Using default parameters.")
                return default_params

    logger.warning(f"Parameters file '{params_path}' not found. Using default parameters.")
    return default_params


def dynamic_param_tuning(best_params, gradient):
    """
    Dynamically adjusts parameters based on performance gradient.

    Args:
        best_params (dict): Current best parameters.
        gradient (dict): Gradient direction for parameters.

    Returns:
        dict: Updated parameters.
    """
    updated_params = best_params.copy()
    for param, direction in gradient.items():
        if direction == 'increase':
            updated_params[param] *= 1.2 
        elif direction == 'decrease':
            updated_params[param] *= 0.8  
        if param == 'learning_rate':
            updated_params[param] = max(0.0001, min(0.01, updated_params[param]))
        elif param == 'batch_size':
            updated_params[param] = max(16, min(128, int(updated_params[param])))
        elif param == 'num_units':
            updated_params[param] = max(32, min(256, int(updated_params[param])))
    return updated_params

def main():
    logger.info("Loading and preprocessing data")
    processed_file = 'data/processed_coffee_shop_data.csv'
    df = process_data(processed_file, 'data/lstm_output.csv')

    features = ['transaction_qty', 'revenue']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaler_path = os.path.join(MODEL_DIR, 'scaler_lstm.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    seq_length = 10
    target_indices = [0, 1]
    X, y = create_sequences(scaled_data, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    best_params = load_best_params() or {'num_units': 128, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}
    logger.info(f"Starting with best parameters: {best_params}")

    best_rmse = float('inf')
    no_improvement_count = 0

    for iteration in range(10):  # Max 10 iterations
        logger.info(f"Iteration {iteration + 1}: Testing parameters {best_params}")
        model, mae, rmse, val_loss = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            best_params['num_units'], best_params['batch_size'], best_params['epochs'],
            best_params['learning_rate'], seq_length, len(target_indices)
        )
        logger.info(f"Results: MAE={mae:.2f}, RMSE={rmse:.2f}, Validation Loss={val_loss:.4f}")

        if rmse < best_rmse:
            logger.info(f"New best RMSE found: {rmse:.2f}")
            best_rmse = rmse

            model.save(os.path.join(MODEL_DIR, 'best_lstm_model.keras'))
            save_best_params(best_params)
            logger.info(f"Updated best parameters: {best_params}")

            # Adjust gradient to tune parameters
            gradient = {'num_units': 'increase', 'batch_size': 'decrease', 'learning_rate': 'decrease'}
            no_improvement_count = 0
        else:
            logger.info("No improvement in RMSE.")
            gradient = {'num_units': 'decrease', 'batch_size': 'increase', 'learning_rate': 'increase'}
            no_improvement_count += 1

        best_params = dynamic_param_tuning(best_params, gradient)

        if no_improvement_count >= 3:
            logger.info("No improvement for 3 iterations. Stopping early.")
            break

if __name__ == "__main__":
    main()