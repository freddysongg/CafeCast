import warnings

import joblib
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import logging
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
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
log_filename = os.path.join(LOG_DIR, f"lstm_bayesian_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging with both console and file handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

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

@tf.function(reduce_retracing=True)
def make_prediction(model, X_test):
    return model(X_test, training=False)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, SEQ_LENGTH, scaler):
    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH, 1)))
    model.add(LSTM(units=int(num_units), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model and capture the training history
    history = model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.1, verbose=0)

    # Log training and validation loss
    final_training_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    logger.info(f"Final Training Loss: {final_training_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}")

    # Evaluate the model
    predictions = make_prediction(model, tf.convert_to_tensor(X_test, dtype=tf.float32))
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    return model, mae, rmse, final_training_loss, final_val_loss

def cross_val_rmse(X, y, num_units, batch_size, epochs, learning_rate, SEQ_LENGTH, scaler, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = Sequential()
        model.add(Input(shape=(SEQ_LENGTH, 1)))
        model.add(LSTM(units=int(num_units), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

        predictions = make_prediction(model, tf.convert_to_tensor(X_val, dtype=tf.float32)).numpy()
        predictions = scaler.inverse_transform(predictions)
        y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_val_actual, predictions))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

def bayesian_optimize(X, y, SEQ_LENGTH, scaler):
    def lstm_model_optimizer(num_units, batch_size, epochs):
        rmse = cross_val_rmse(
            X, y, 
            num_units=int(num_units), 
            batch_size=int(batch_size), 
            epochs=int(epochs), 
            learning_rate=0.001, 
            SEQ_LENGTH=SEQ_LENGTH, 
            scaler=scaler
        )
        return -rmse  # Negative because Bayesian optimization maximizes the function

    pbounds = {
        'num_units': (50, 300),
        'batch_size': (8, 128),
        'epochs': (50, 300)
    }

    optimizer = BayesianOptimization(
        f=lstm_model_optimizer,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=10, n_iter=50)

    best_params = optimizer.max['params']
    logger.info(f"Best Parameters found - num_units: {int(best_params['num_units'])}, "
                f"batch_size: {int(best_params['batch_size'])}, epochs: {int(best_params['epochs'])}")
    logger.info(f"Best RMSE: {-optimizer.max['target']:.2f}")

    # Save best parameters
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_params.json')
    best_params_converted = {key: int(value) for key, value in best_params.items()}
    with open(params_path, 'w') as f:
        json.dump(best_params_converted, f, indent=4)

    return best_params_converted

def load_best_model_metrics():
    metrics_path = os.path.join(MODEL_DIR, 'best_lstm_bayesian_model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def save_best_model_metrics(metrics):
    metrics_path = os.path.join(MODEL_DIR, 'best_lstm_bayesian_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Best model metrics saved to {metrics_path}")

def remove_existing_model_if_better(new_rmse):
    best_metrics = load_best_model_metrics()
    if best_metrics is not None:
        best_rmse = best_metrics.get('rmse', float('inf'))
        if new_rmse >= best_rmse:
            logger.info(f"New model RMSE ({new_rmse:.2f}) is not better than existing best RMSE ({best_rmse:.2f}). Keeping the old model.")
            return False
    # If no existing metrics or new model is better, delete the existing model
    model_path = os.path.join(MODEL_DIR, 'best_lstm_bayesian_model.keras')
    if os.path.exists(model_path):
        os.remove(model_path)
        logger.info(f"Deleted old model file: {model_path}")
    return True

def save_scaler(scaler, path="models/scaler.pkl"):
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")
    
def load_scaler(path="models/scaler.pkl"):
    if os.path.exists(path):
        logger.info(f"Scaler loaded from {path}")
        return joblib.load(path)
    else:
        logger.error(f"No scaler found at {path}. Ensure the scaler is saved during training.")
        return None
            

def main():
    logger.info("Loading and preparing data")
    data = prepare_data('data/cafecast_data.xlsx')
    daily_data = data.resample('D')['transaction_qty'].sum()
    logger.info(f"Data overview: {daily_data.describe()}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))
    save_scaler(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    SEQ_LENGTH = 10
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")
    logger.info("Starting Bayesian Optimization")

    best_params = bayesian_optimize(X_train, y_train, SEQ_LENGTH, scaler)

    # Train final model with best parameters found
    logger.info(f"Training final model with best parameters: {best_params}")
    final_model, mae, rmse, training_loss, val_loss = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        num_units=int(best_params['num_units']),
        batch_size=int(best_params['batch_size']),
        epochs=int(best_params['epochs']),
        learning_rate=0.001,
        SEQ_LENGTH=SEQ_LENGTH,
        scaler=scaler
    )

    # Compare the new model with the existing best model
    if remove_existing_model_if_better(rmse):
        model_path = os.path.join(MODEL_DIR, 'best_lstm_bayesian_model.keras')
        final_model.save(model_path)
        logger.info(f"New best model saved to {model_path}")
        save_best_model_metrics({'mae': mae, 'rmse': rmse, 'training_loss': training_loss, 'val_loss': val_loss})
    else:
        logger.info("New model was not better. Existing best model retained.")

if __name__ == "__main__":
    main()