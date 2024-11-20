import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import glob
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data.modify_dataset import prepare_data

# Ensure the logs directory exists
LOG_DIR = 'logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a timestamped log file name
log_filename = os.path.join(LOG_DIR, f"transformer_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging with both console and file handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a console handler for real-time terminal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Create a file handler for saving logs at the end
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

def main():
    # Clean up old logs at the start
    cleanup_old_logs(LOG_DIR)

    # Load and prepare data
    logger.info("Loading and preparing data")
    data = prepare_data('data/cafecast_data.xlsx')
    daily_data = data.resample('D')['transaction_qty'].sum()

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))

    # Create sequences for Transformer input
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 10  # Base sequence length
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).squeeze()
    y_test = torch.FloatTensor(y_test).squeeze()

    # Initialize best parameters
    best_params = {
        'num_layers': 2,
        'num_heads': 2,
        'd_model': 32,
        'dim_feedforward': 64,
        'learning_rate': 0.001
    }
    best_rmse = float('inf')

    iteration = 1
    while best_rmse > 200:
        logger.info(f"Iteration {iteration} starting with parameters: {best_params}")

        # Ensure d_model is divisible by num_heads
        d_model = best_params['d_model']
        num_heads = best_params['num_heads']
        if d_model % num_heads != 0:
            num_heads = max(1, d_model // (d_model // num_heads))
            logger.info(f"Adjusted num_heads to {num_heads} to ensure d_model ({d_model}) is divisible by num_heads")

        X_train_expanded = X_train.expand(-1, -1, d_model)
        X_test_expanded = X_test.expand(-1, -1, d_model)

        # Define the Transformer model
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

        model = TimeSeriesTransformer(
            input_size=d_model,
            num_layers=best_params['num_layers'],
            num_heads=num_heads,
            d_model=d_model,
            dim_feedforward=best_params['dim_feedforward']
        )

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        # Training loop
        epochs = 50
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_expanded)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Iteration {iteration}, Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_expanded).squeeze().numpy()
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

        # Evaluate the model
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        logger.info(f"Iteration {iteration} - Evaluation results - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Check for the best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {
                'num_layers': best_params['num_layers'] + 1 if best_params['num_layers'] < 4 else best_params['num_layers'],
                'num_heads': min(num_heads + 2, 8),
                'd_model': min(best_params['d_model'] * 2, 128),
                'dim_feedforward': min(best_params['dim_feedforward'] * 2, 256),
                'learning_rate': max(best_params['learning_rate'] / 2, 0.0001)
            }
            logger.info(f"New best model found: {best_params} with RMSE: {rmse:.2f}")

        iteration += 1

    # Log the best model parameters
    logger.info(f"Final best model parameters: {best_params} with RMSE: {best_rmse:.2f}")

if __name__ == "__main__":
    main()
