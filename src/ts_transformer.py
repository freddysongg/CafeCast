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
from tensorflow.keras.layers import Embedding, Concatenate, LayerNormalization, GlobalAveragePooling1D, Dropout, Dense, Input # type: ignore
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

def build_transformer_model(seq_length, num_features, num_heads, num_layers, d_model, ff_dim, target_size, num_products, dropout_rate):
    """
    Builds a Transformer-based model for time-series forecasting with product embeddings.

    Args:
        seq_length (int): Length of input sequences.
        num_features (int): Number of numerical input features.
        num_heads (int): Number of attention heads in the Transformer layers.
        num_layers (int): Number of Transformer layers.
        d_model (int): Dimension of embeddings for both product IDs and numerical inputs.
        ff_dim (int): Dimension of the feed-forward layers.
        target_size (int): Number of output targets (e.g., predictions per time step).
        num_products (int): Number of unique product IDs for embedding.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: A compiled Transformer-based time-series forecasting model.
    """
    product_input = Input(shape=(seq_length,), name='product_id')
    product_embedding = Embedding(input_dim=num_products, output_dim=d_model)(product_input)

    numerical_input = Input(shape=(seq_length, num_features), name='numerical_features')
    numerical_projection = Dense(d_model)(numerical_input) 

    combined = Concatenate()([numerical_projection, product_embedding])
    combined = Dense(d_model)(combined)  
    x = LayerNormalization()(combined)

    for _ in range(num_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization()(x + attention_output)

        feed_forward_output = Dense(ff_dim, activation='relu')(x)
        feed_forward_output = Dense(d_model)(feed_forward_output)  
        x = LayerNormalization()(x + feed_forward_output)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(target_size)(x)

    model = Model(inputs=[product_input, numerical_input], outputs=outputs)
    return model

def train_and_evaluate_model(inputs_train, inputs_test, y_train, y_test, num_heads, num_layers, d_model, ff_dim, learning_rate, dropout_rate, seq_length, target_size):
    """
    Trains and evaluates the Transformer model for time-series forecasting.

    Args:
        inputs_train (list): Training inputs, including product IDs and numerical features ([product_train, X_train]).
        inputs_test (list): Testing inputs, including product IDs and numerical features ([product_test, X_test]).
        y_train (np.array): Training target outputs of shape (num_samples, target_size).
        y_test (np.array): Testing target outputs of shape (num_samples, target_size).
        num_heads (int): Number of attention heads in the Transformer layers.
        num_layers (int): Number of Transformer layers.
        d_model (int): Dimension of embeddings for inputs.
        ff_dim (int): Dimension of the feed-forward layers.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for regularization.
        seq_length (int): Length of input sequences.
        target_size (int): Number of output targets.

    Returns:
        tf.keras.Model: The trained Transformer model.
        float: Root Mean Squared Error (RMSE) on the test dataset.
        float: Final validation loss from the training history.
    """
    product_train, X_train = inputs_train
    product_test, X_test = inputs_test

    num_features = X_train.shape[2]
    num_products = product_train.max() + 1  

    model = build_transformer_model(seq_length, num_features, num_heads, num_layers, d_model, ff_dim, target_size, num_products, dropout_rate)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    history = model.fit([product_train, X_train], y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    predictions = model.predict([product_test, X_test])
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, rmse, history.history['val_loss'][-1]

def dynamic_param_tuning(best_params, gradient):
    """
    Dynamically adjusts hyperparameters based on a gradient or exploration strategy.

    Args:
        best_params (dict): Current best hyperparameters.
        gradient (dict): Dictionary indicating parameter adjustment directions (e.g., 'increase', 'decrease').

    Returns:
        dict: Updated hyperparameters.
    """
    updated_params = best_params.copy()
    for param, direction in gradient.items():
        if direction == 'increase':
            updated_params[param] *= 1.2 
        elif direction == 'decrease':
            updated_params[param] *= 0.8

        if param == 'learning_rate':
            updated_params[param] = max(0.0001, min(0.01, updated_params[param]))
        elif param == 'num_heads':
            updated_params[param] = max(1, min(16, int(updated_params[param])))
        elif param == 'num_layers':
            updated_params[param] = max(1, min(10, int(updated_params[param])))
        elif param == 'd_model':
            updated_params[param] = max(16, min(512, int(updated_params[param])))
        elif param == 'ff_dim':
            updated_params[param] = max(32, min(2048, int(updated_params[param])))
        elif param == 'dropout_rate':
            updated_params[param] = max(0.0, min(0.5, updated_params[param]))

    return updated_params

def save_best_params(best_params):
    """
    Saves the best hyperparameters dynamically to a JSON file.

    Args:
        best_params (dict): Dictionary containing the best hyperparameters. 
                            Keys must include:
                            - 'num_heads'
                            - 'num_layers'
                            - 'd_model'
                            - 'ff_dim'
                            - 'learning_rate'
                            - 'dropout_rate'.

    Notes:
        If any required keys are missing, default values are added before saving.
        The parameters are saved to `params/best_ts_transformer_params.json`.
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
    Loads the most recent hyperparameters from a JSON file.

    Returns:
        dict: A dictionary containing the loaded hyperparameters. If the file doesn't exist or keys are missing, 
              default values are used.

    Notes:
        Default parameters include:
        - num_heads: 4
        - num_layers: 2
        - d_model: 64
        - ff_dim: 128
        - learning_rate: 0.001
        - dropout_rate: 0.1
    """
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')

    default_params = {'num_heads': 4, 'num_layers': 2, 'd_model': 64, 'ff_dim': 128, 'learning_rate': 0.001, 'dropout_rate': 0.1}

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)

            if isinstance(params, list):
                if len(params) == 0:
                    logger.warning(f"Parameters file '{params_path}' is empty. Using default parameters.")
                    return default_params
                params = params[-1]  

            for key, default_value in default_params.items():
                if key not in params:
                    logger.warning(f"Key '{key}' missing in loaded params. Adding default value: {default_value}")
                    params[key] = default_value

            return params

    logger.warning(f"Parameters file '{params_path}' not found. Using default parameters.")
    return default_params

def main():
    """
    Main function to preprocess data, build, train, and evaluate the Transformer model.

    Steps:
        1. Loads and preprocesses the dataset.
        2. Maps product IDs to integers for embedding.
        3. Splits the data into training and testing sets.
        4. Loads the best hyperparameters (if available) or initializes default values.
        5. Iteratively trains and evaluates the Transformer model with the current parameters.
        6. Saves the trained model and the best parameters to disk.

    Notes:
        - The process stops early if no improvement is observed for 3 consecutive iterations.
        - Saves the model to `models/best_ts_transformer_model.keras`.
        - Saves the best hyperparameters to `params/best_ts_transformer_params.json`.
    """
    logger.info("Loading and preprocessing data")
    processed_file = 'data/processed_coffee_shop_data.csv'
    df = process_data(processed_file, 'data/ts_transformer_output.csv')

    unique_products = df['product_id'].unique()
    product_mapping = {product: idx for idx, product in enumerate(unique_products)}
    df['product_id'] = df['product_id'].map(product_mapping)

    features = ['transaction_qty', 'revenue'] + [col for col in df.columns if col.startswith('dow_') or col.startswith('month_')]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    scaler_path = os.path.join(MODEL_DIR, 'scaler_ts_transformer.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    seq_length = 10
    target_indices = [0, 1]
    product_sequences, X, y = create_sequences(scaled_data, df['product_id'].values, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    product_train, product_test = product_sequences[:train_size], product_sequences[train_size:]

    best_params = load_best_params()
    logger.info(f"Starting with best parameters: {best_params}")

    best_rmse = float('inf')
    no_improvement_count = 0

    for iteration in range(10):  # Max 10 iterations
        logger.info(f"Iteration {iteration + 1}: Testing parameters {best_params}")
        model, rmse, val_loss = train_and_evaluate_model(
            [product_train, X_train], [product_test, X_test],
            y_train, y_test, best_params['num_heads'], best_params['num_layers'],
            best_params['d_model'], best_params['ff_dim'], best_params['learning_rate'],
            best_params['dropout_rate'], seq_length, len(target_indices)
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

        gradient = {'learning_rate': 'increase', 'd_model': 'decrease'}  
        best_params = dynamic_param_tuning(best_params, gradient)

    logger.info(f"Final RMSE after tuning: {best_rmse:.2f}")

if __name__ == "__main__":
    main()
