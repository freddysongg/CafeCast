import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import logging
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from data.modify_dataset import prepare_data

# Ensure the logs, models, and params directories exist
LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

# Generate a timestamped log file name
log_filename = os.path.join(LOG_DIR, f"lstm_bayesian_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler for real-time output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler for logs
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, SEQ_LENGTH, scaler):
    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH, 1)))
    model.add(LSTM(units=int(num_units), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    # Evaluate the model
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    logger.info(f"Model evaluation - num_units: {num_units}, batch_size: {batch_size}, epochs: {epochs}")
    logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return rmse  # We return RMSE as it is the optimization target

# Load data
logger.info("Loading and preparing data")
data = prepare_data('data/cafecast_data.xlsx')
daily_data = data.resample('D')['transaction_qty'].sum()
logger.info(f"Data overview: {daily_data.describe()}")

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))
SEQ_LENGTH = 10
X, y = create_sequences(scaled_data, SEQ_LENGTH)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

# Define the search space for Bayesian Optimization
search_space = [
    Integer(50, 300, name='num_units'),
    Integer(16, 128, name='batch_size'),
    Integer(50, 300, name='epochs')
]

@use_named_args(search_space)
def objective_function(num_units, batch_size, epochs):
    # Run the model training and evaluation
    rmse = train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, 0.001, SEQ_LENGTH, scaler)
    return rmse

# Run Bayesian Optimization
logger.info("Starting Bayesian Optimization")
result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=20,  # Number of total iterations
    random_state=42
)

# Log best result
best_num_units = result.x[0]
best_batch_size = result.x[1]
best_epochs = result.x[2]
logger.info(f"Best Parameters found - num_units: {best_num_units}, batch_size: {best_batch_size}, epochs: {best_epochs}")
logger.info(f"Best RMSE: {result.fun:.2f}")

# Save the best parameters
best_params = {
    'num_units': best_num_units,
    'batch_size': best_batch_size,
    'epochs': best_epochs
}
params_path = os.path.join(PARAMS_DIR, 'best_lstm_bayesian_params.json')
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=4)
logger.info(f"Best parameters saved to {params_path}")

# Remove existing model before saving the new best model
def remove_existing_model(model_dir):
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path) and file_path.endswith('.keras'):
            os.remove(file_path)
            logger.info(f"Deleted old model file: {file_path}")

# Save the final best model
logger.info("Saving the final model with best parameters")
remove_existing_model(MODEL_DIR)
final_model, _, _, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test, best_num_units, best_batch_size, best_epochs, 0.001, SEQ_LENGTH, scaler)
model_path = os.path.join(MODEL_DIR, 'best_lstm_bayesian_model.keras')
final_model.save(model_path)
logger.info(f"Best model saved to {model_path}")
