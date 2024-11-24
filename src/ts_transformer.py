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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay # type: ignore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from preprocess_data import process_data

LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"ts_transformer_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
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

def build_transformer_model(seq_length, num_features, num_heads, num_layers, d_model, ff_dim, target_size, dropout_rate):
    """
    Builds a Transformer model for time-series forecasting.

    Args:
        seq_length (int): Length of input sequences.
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer layers.
        d_model (int): Embedding dimension.
        ff_dim (int): Feedforward dimension.
        target_size (int): Number of output targets.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: Compiled Transformer model.
    """
    inputs = Input(shape=(seq_length, num_features))
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)

    for _ in range(num_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attention_output)
        feed_forward_output = Dense(ff_dim, activation='relu')(x)
        x = LayerNormalization()(x + feed_forward_output)

    x = Dropout(dropout_rate)(x)
    x = tf.reduce_mean(x, axis=1)  # Global average pooling
    outputs = Dense(target_size)(x)

    model = Model(inputs, outputs)
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, num_heads, num_layers, d_model, ff_dim, learning_rate, dropout_rate, seq_length, target_size):
    """
    Trains and evaluates the Transformer model.

    Args:
        Various model hyperparameters.

    Returns:
        float: RMSE of the model.
    """
    num_features = X_train.shape[2]
    model = build_transformer_model(seq_length, num_features, num_heads, num_layers, d_model, ff_dim, target_size, dropout_rate)
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return model, rmse, history.history['val_loss'][-1]

def save_best_params(best_params):
    """
    Dynamically saves the best parameters.

    Args:
        best_params (dict): The best parameters to save.
    """
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')

    required_keys = ['num_heads', 'num_layers', 'd_model', 'ff_dim', 'learning_rate', 'dropout_rate']
    for key in required_keys:
        if key not in best_params:
            logger.warning(f"Missing parameter '{key}' in best_params before saving. Adding default value.")
            if key == 'learning_rate':
                best_params[key] = 0.001
            elif key == 'num_heads':
                best_params[key] = 4
            elif key == 'num_layers':
                best_params[key] = 2
            elif key == 'd_model':
                best_params[key] = 64
            elif key == 'ff_dim':
                best_params[key] = 128
            elif key == 'dropout_rate':
                best_params[key] = 0.1

    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Best parameters saved dynamically to {params_path}")

def load_best_params():
    """
    Loads the best parameters from a JSON file. If some keys are missing,
    default values are added.

    Returns:
        dict: Best parameters with all required keys.
    """
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')

    default_params = {'num_heads': 4, 'num_layers': 2, 'd_model': 64, 'ff_dim': 128, 'learning_rate': 0.001, 'dropout_rate': 0.1}

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)

            for key, default_value in default_params.items():
                if key not in params:
                    logger.warning(f"Key '{key}' missing in loaded params. Adding default value: {default_value}")
                    params[key] = default_value

            return params

    logger.warning(f"Parameters file '{params_path}' not found. Using default parameters.")
    return default_params

def main():
    logger.info("Loading and preprocessing data")
    processed_file = 'data/processed_coffee_shop_data.csv'
    df = process_data(processed_file, 'data/ts_transformer_output.csv')

    features = ['transaction_qty', 'revenue']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaler_path = os.path.join(MODEL_DIR, 'scaler_ts_transformer.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    seq_length = 10
    target_indices = [0, 1]
    X, y = create_sequences(scaled_data, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    best_params = load_best_params()
    logger.info(f"Starting with best parameters: {best_params}")

    best_rmse = float('inf')
    no_improvement_count = 0

    for iteration in range(10):  # Max 10 iterations
        logger.info(f"Iteration {iteration + 1}: Testing parameters {best_params}")
        model, rmse, val_loss = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            best_params['num_heads'], best_params['num_layers'], best_params['d_model'],
            best_params['ff_dim'], best_params['learning_rate'], best_params['dropout_rate'],
            seq_length, len(target_indices)
        )
        logger.info(f"Results: RMSE={rmse:.2f}, Validation Loss={val_loss:.4f}")

        if rmse < best_rmse:
            logger.info(f"New best RMSE found: {rmse:.2f}")
            best_rmse = rmse
            model.save(os.path.join(MODEL_DIR, 'best_ts_transformer_model.keras'))
            save_best_params(best_params)
            no_improvement_count = 0
        else:
            logger.info("No improvement in RMSE.")
            no_improvement_count += 1

        if no_improvement_count >= 3:
            logger.info("No improvement for 3 iterations. Stopping early.")
            break

if __name__ == "__main__":
    main()
