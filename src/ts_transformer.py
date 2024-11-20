import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import glob
import logging
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from data.modify_dataset import prepare_data

# Ensure the logs, models, and params directories exist
LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Generate a timestamped log file name
log_filename = os.path.join(LOG_DIR, f"transformer_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

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
def cleanup_old_logs(directory, prefix="transformer_log_", max_logs=5):
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

# Transformer model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, d_model, dim_feedforward):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Get the output of the last time step
        return x

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_layers, num_heads, d_model, dim_feedforward, learning_rate, scaler):
    model = TimeSeriesTransformer(
        input_size=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        dim_feedforward=dim_feedforward
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure the input data is projected to the d_model dimension
    X_train = torch.FloatTensor(X_train).reshape(-1, X_train.shape[1], 1)  # (batch_size, seq_length, 1)
    X_test = torch.FloatTensor(X_test).reshape(-1, X_test.shape[1], 1)

    # Create a linear projection layer if necessary
    projection = nn.Linear(1, d_model)
    X_train = projection(X_train)
    X_test = projection(X_test)

    # Training loop
    model.train()
    epochs = 50  # Keeping epochs constant for parameter tuning
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), torch.FloatTensor(y_train))
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    return model, mae, rmse, loss.item()


def plot_metrics(metrics_history, param_name):
    param_values = [entry[param_name] for entry in metrics_history]
    mae_values = [entry['mae'] for entry in metrics_history]
    rmse_values = [entry['rmse'] for entry in metrics_history]
    val_loss_values = [entry['val_loss'] for entry in metrics_history]

    plt.figure(figsize=(14, 7))
    plt.plot(param_values, mae_values, marker='o', label='MAE')
    plt.plot(param_values, rmse_values, marker='o', label='RMSE')
    plt.plot(param_values, val_loss_values, marker='o', label='Validation Loss')
    plt.xlabel(param_name.capitalize())
    plt.ylabel('Metric Value')
    plt.title(f'Metrics vs {param_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_best_params(best_params):
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')
    
    # Ensure all values are native Python types for JSON serialization
    best_params_converted = {key: int(value) if isinstance(value, (np.integer, np.int64)) else value
                             for key, value in best_params.items()}
    
    # Check if the file exists and load existing parameters
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            existing_params = json.load(f)
        if isinstance(existing_params, dict):
            existing_params = [existing_params]
    else:
        existing_params = []

    existing_params.append(best_params_converted)

    # Save the updated parameter history
    with open(params_path, 'w') as f:
        json.dump(existing_params, f, indent=4)
    logger.info(f"Best parameters updated and saved to {params_path}")

def load_best_params():
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params_list = json.load(f)
            if isinstance(params_list, list) and params_list:
                return params_list[-1]  # Return the most recent best parameters
    return None

def clear_params():
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')
    if os.path.exists(params_path):
        os.remove(params_path)
        logger.info(f"Cleared parameter file: {params_path}")

def remove_existing_model(model_dir):
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path) and file_path.endswith('.pt'):
            os.remove(file_path)
            logger.info(f"Deleted old model file: {file_path}")

def main():
    cleanup_old_logs(LOG_DIR)

    # Option to clear saved parameters
    if len(sys.argv) > 1 and sys.argv[1] == 'clear_params':
        clear_params()
        return

    logger.info("Starting Transformer model script")

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
    best_params = load_best_params() or {'num_layers': 2, 'num_heads': 2, 'd_model': 64, 'dim_feedforward': 128}
    logger.info(f"Starting with initial best parameters: {best_params}")

    # Step 1: Find the best num_layers
    num_layers_list = [max(1, best_params['num_layers'] + delta) for delta in np.linspace(-1, 2, 5, dtype=int)]
    metrics_history = []

    for num_layers in num_layers_list:
        logger.info(f"Testing num_layers={num_layers}")
        model, mae, rmse, val_loss = train_and_evaluate_model(X_train, X_test, y_train, y_test, num_layers, best_params['num_heads'], best_params['d_model'], best_params['dim_feedforward'], 0.001, scaler)
        metrics_history.append({'num_layers': num_layers, 'num_heads': best_params['num_heads'], 'd_model': best_params['d_model'], 'dim_feedforward': best_params['dim_feedforward'], 'mae': mae, 'rmse': rmse, 'val_loss': val_loss})
        logger.info(f"num_layers={num_layers} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Final Val Loss: {val_loss:.4f}")

    # Plot and find the best num_layers
    plot_metrics(metrics_history, 'num_layers')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['num_layers'] = best_entry['num_layers']
    logger.info(f"Best num_layers found: {best_params['num_layers']} with RMSE: {best_entry['rmse']:.2f}")

    # Step 2: Find the best num_heads
    num_heads_list = [max(1, best_params['num_heads'] + delta) for delta in np.linspace(-1, 2, 5, dtype=int)]
    metrics_history = []  # Reset metrics history

    for num_heads in num_heads_list:
        logger.info(f"Testing num_heads={num_heads}")
        model, mae, rmse, val_loss = train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params['num_layers'], num_heads, best_params['d_model'], best_params['dim_feedforward'], 0.001, scaler)
        metrics_history.append({'num_layers': best_params['num_layers'], 'num_heads': num_heads, 'd_model': best_params['d_model'], 'dim_feedforward': best_params['dim_feedforward'], 'mae': mae, 'rmse': rmse, 'val_loss': val_loss})
        logger.info(f"num_heads={num_heads} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Final Val Loss: {val_loss:.4f}")

    # Plot and find the best num_heads
    plot_metrics(metrics_history, 'num_heads')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['num_heads'] = best_entry['num_heads']
    logger.info(f"Best num_heads found: {best_params['num_heads']} with RMSE: {best_entry['rmse']:.2f}")

    # Step 3: Find the best d_model
    d_model_list = [max(32, best_params['d_model'] + delta) for delta in np.linspace(-16, 32, 5, dtype=int)]
    metrics_history = []  # Reset metrics history

    for d_model in d_model_list:
        logger.info(f"Testing d_model={d_model}")
        model, mae, rmse, val_loss = train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params['num_layers'], best_params['num_heads'], d_model, best_params['dim_feedforward'], 0.001, scaler)
        metrics_history.append({'num_layers': best_params['num_layers'], 'num_heads': best_params['num_heads'], 'd_model': d_model, 'dim_feedforward': best_params['dim_feedforward'], 'mae': mae, 'rmse': rmse, 'val_loss': val_loss})
        logger.info(f"d_model={d_model} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Final Val Loss: {val_loss:.4f}")

    # Plot and find the best d_model
    plot_metrics(metrics_history, 'd_model')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['d_model'] = best_entry['d_model']
    logger.info(f"Best d_model found: {best_params['d_model']} with RMSE: {best_entry['rmse']:.2f}")

    # Step 4: Find the best dim_feedforward
    dim_feedforward_list = [max(64, best_params['dim_feedforward'] + delta) for delta in np.linspace(-64, 128, 5, dtype=int)]
    metrics_history = []  # Reset metrics history

    for dim_feedforward in dim_feedforward_list:
        logger.info(f"Testing dim_feedforward={dim_feedforward}")
        model, mae, rmse, val_loss = train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params['num_layers'], best_params['num_heads'], best_params['d_model'], dim_feedforward, 0.001, scaler)
        metrics_history.append({'num_layers': best_params['num_layers'], 'num_heads': best_params['num_heads'], 'd_model': best_params['d_model'], 'dim_feedforward': dim_feedforward, 'mae': mae, 'rmse': rmse, 'val_loss': val_loss})
        logger.info(f"dim_feedforward={dim_feedforward} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Final Val Loss: {val_loss:.4f}")

    # Plot and find the best dim_feedforward
    plot_metrics(metrics_history, 'dim_feedforward')
    best_entry = min(metrics_history, key=lambda x: x['rmse'])
    best_params['dim_feedforward'] = best_entry['dim_feedforward']
    logger.info(f"Best dim_feedforward found: {best_params['dim_feedforward']} with RMSE: {best_entry['rmse']:.2f}")

    # Save the updated best parameters
    save_best_params(best_params)

    # Remove existing model before saving the new best model
    remove_existing_model(MODEL_DIR)

    # Save the final best model
    model_path = os.path.join(MODEL_DIR, 'best_ts_transformer_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
