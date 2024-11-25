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
import torch
import torch.nn as nn
import torch.optim as optim

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

def create_sequences(data, product_ids, seq_length, target_indices):
    """
    Creates input-output sequences for time-series forecasting, along with product-level inputs.

    Args:
        data (np.array): Array of shape (time_steps, features). Contains the numerical features.
        product_ids (np.array): Array of product IDs corresponding to each time step.
        seq_length (int): Length of each input sequence.
        target_indices (list): Indices of target columns in the data to be predicted.

    Returns:
        np.array: Product ID sequences of shape (num_sequences, seq_length).
        np.array: Numerical input sequences of shape (num_sequences, seq_length, features).
        np.array: Target outputs of shape (num_sequences, len(target_indices)).
    """
    X, y, products = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_indices])
        products.append(product_ids[i:i + seq_length])
    return np.array(products), np.array(X), np.array(y)

class LSTMModel(nn.Module):
    """
    Defines a PyTorch-based LSTM model for time-series forecasting.

    Args:
        input_size (int): Number of numerical input features per time step.
        num_units (int): Number of hidden units in the LSTM layer.
        target_size (int): Number of output targets (e.g., dimensions of the predictions).

    Methods:
        forward(x):
            Processes input data through the LSTM layer and the fully connected layer.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            Returns:
                torch.Tensor: Predictions of shape (batch_size, target_size).
    """
    def __init__(self, input_size, num_units, target_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, num_units, batch_first=True)
        self.fc = nn.Linear(num_units, target_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, seq_length, target_size, device):
    """
    Trains and evaluates the LSTM model using PyTorch.

    Args:
        X_train (np.array): Training input data of shape (num_samples, seq_length, num_features).
        X_test (np.array): Testing input data of shape (num_samples, seq_length, num_features).
        y_train (np.array): Training target data of shape (num_samples, target_size).
        y_test (np.array): Testing target data of shape (num_samples, target_size).
        num_units (int): Number of hidden units in the LSTM layer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        seq_length (int): Sequence length for each input sample.
        target_size (int): Number of output targets.
        device (torch.device): The device to run the model on ('cuda' for GPU or 'cpu').

    Returns:
        LSTMModel: The trained PyTorch model.
        float: Mean Absolute Error (MAE) on the test data.
        float: Root Mean Squared Error (RMSE) on the test data.
    """
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, num_units, target_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        predictions = model(X_test_tensor).cpu().numpy()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    return model, mae, rmse

def save_best_params(best_params):
    """
    Saves the best hyperparameters dynamically to a JSON file. Appends to an existing file if present.

    Args:
        best_params (dict): A dictionary containing the best hyperparameters. 
                            Keys must include 'num_units', 'batch_size', 'epochs', and 'learning_rate'.

    Notes:
        If required keys are missing, default values are added before saving.
        The parameters are appended to a list if a JSON file already exists.
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
    Loads the most recent hyperparameters from a JSON file.

    Returns:
        dict: A dictionary containing the loaded hyperparameters. If the file doesn't exist
              or keys are missing, default values are used.

    Notes:
        Default parameters include:
        - num_units: 128
        - batch_size: 32
        - epochs: 50
        - learning_rate: 0.001
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
    Dynamically adjusts hyperparameters based on performance gradients.

    Args:
        best_params (dict): Current best hyperparameters (e.g., num_units, batch_size, etc.).
        gradient (dict): A dictionary indicating whether to 'increase' or 'decrease' each parameter.

    Returns:
        dict: Updated hyperparameters after applying the gradient adjustments.

    Notes:
        - 'learning_rate' is clamped between 0.0001 and 0.01.
        - 'batch_size' is clamped between 16 and 128.
        - 'num_units' is clamped between 32 and 256.
        - Gradients are applied multiplicatively:
            - 'increase': Parameter is multiplied by 1.2.
            - 'decrease': Parameter is multiplied by 0.8.
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
    """
    Main function to run the LSTM training and evaluation pipeline.

    Steps:
        1. Loads and preprocesses the dataset.
        2. Splits the data into training and testing sets.
        3. Loads the best hyperparameters (if available) or initializes defaults.
        4. Iteratively trains the LSTM model with the current parameters.
        5. Dynamically tunes hyperparameters based on performance gradients.
        6. Saves the best hyperparameters and the trained model.

    Notes:
        - The process stops early if no improvement is observed for 3 consecutive iterations.
        - The model is saved to `models/best_lstm_model.pth`.
        - The best hyperparameters are saved to `params/best_lstm_params.json`.
    """
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_params = load_best_params() or {'num_units': 128, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}
    logger.info(f"Starting with best parameters: {best_params}")

    best_rmse = float('inf')
    no_improvement_count = 0

    for iteration in range(10):  # Max 10 iterations
        logger.info(f"Iteration {iteration + 1}: Testing parameters {best_params}")
        model, mae, rmse = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            best_params['num_units'], best_params['batch_size'], best_params['epochs'],
            best_params['learning_rate'], seq_length, len(target_indices), device
        )
        logger.info(f"Results: MAE={mae:.2f}, RMSE={rmse:.2f}")

        if rmse < best_rmse:
            logger.info(f"New best RMSE found: {rmse:.2f}")
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_lstm_model.pth'))
            save_best_params(best_params)
            logger.info(f"Updated best parameters: {best_params}")
            no_improvement_count = 0
        else:
            logger.info("No improvement in RMSE.")
            no_improvement_count += 1

        if no_improvement_count >= 3:
            logger.info("No improvement for 3 iterations. Stopping early.")
            break

if __name__ == "__main__":
    main()