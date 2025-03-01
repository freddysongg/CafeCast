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
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate # type: ignore
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

def train_and_evaluate_model(product_train, X_train, product_test, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, seq_length, target_size, num_products, embedding_dim):
    """
    Trains and evaluates a TensorFlow LSTM model with product embeddings.

    Args:
        product_train (np.ndarray): Training product IDs (sequences of product IDs).
        X_train (np.ndarray): Training numerical input features.
        product_test (np.ndarray): Testing product IDs (sequences of product IDs).
        X_test (np.ndarray): Testing numerical input features.
        y_train (np.ndarray): Training target outputs.
        y_test (np.ndarray): Testing target outputs.
        num_units (int): Number of hidden units in the LSTM layer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        seq_length (int): Sequence length of the input data.
        target_size (int): Number of output targets (e.g., predictions per time step).
        num_products (int): Number of unique product IDs for embedding.
        embedding_dim (int): Dimension of the product embedding.

    Returns:
        tf.keras.Model: The trained TensorFlow LSTM model.
        float: Mean Absolute Error (MAE) on the test dataset.
        float: Root Mean Squared Error (RMSE) on the test dataset.
    """
    product_input = Input(shape=(seq_length,), name='product_id')
    numerical_input = Input(shape=(seq_length, X_train.shape[2]), name='numerical_features')

    product_embedding = Embedding(input_dim=num_products, output_dim=embedding_dim)(product_input)

    combined_input = Concatenate()([numerical_input, product_embedding])

    lstm_output = LSTM(units=int(num_units), activation='relu')(combined_input)
    outputs = Dense(target_size)(lstm_output)

    model = model(inputs=[product_input, numerical_input], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    model.fit([product_train, X_train], y_train, epochs=int(epochs), batch_size=int(batch_size), validation_split=0.1, verbose=0)

    predictions = model.predict([product_test, X_test])
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return model, mae, rmse

def save_best_params(best_params):
    """
    Saves the best hyperparameters dynamically to a JSON file.

    Args:
        best_params (dict): Dictionary containing the best hyperparameters. 
                            Keys must include 'num_units', 'batch_size', 'epochs', and 'learning_rate'.

    Notes:
        If any required keys are missing, default values are added before saving.
        The parameters are saved as a JSON file (`params/best_lstm_params.json`).
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
        dict: A dictionary containing the loaded hyperparameters. If the file doesn't exist or keys are missing, 
              default values are used.

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
    Main function to preprocess data, build, train, and evaluate the LSTM model.

    Steps:
        1. Loads and preprocesses the dataset.
        2. Maps product IDs to integers for embedding.
        3. Splits the data into training and testing sets.
        4. Initializes hyperparameters for the LSTM model:
           - num_units: Number of LSTM hidden units.
           - batch_size: Batch size for training.
           - epochs: Number of epochs for training.
           - learning_rate: Learning rate for the optimizer.
           - seq_length: Sequence length for inputs.
           - target_size: Number of prediction targets.
           - num_products: Number of unique products for embedding.
           - embedding_dim: Dimension of the product embedding.
        5. Builds, trains, and evaluates the LSTM model.
        6. Saves the trained model to disk.

    Notes:
        - Assumes preprocessed data from `process_data`.
        - Saves the model to `models/best_lstm_model.keras`.
    """
    logger.info("Loading and preprocessing data")
    processed_file = 'data/processed_coffee_shop_data.csv'
    df = process_data(processed_file, 'data/lstm_output.csv')

    unique_products = df['product_id'].unique()
    product_mapping = {product: idx for idx, product in enumerate(unique_products)}
    df['product_id'] = df['product_id'].map(product_mapping)

    features = ['transaction_qty', 'revenue']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    product_ids = df['product_id'].values

    scaler_path = os.path.join(MODEL_DIR, 'scaler_lstm.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    seq_length = 10
    target_indices = [0, 1]
    product_sequences, X, y = create_sequences(scaled_data, product_ids, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    product_train, product_test = product_sequences[:train_size], product_sequences[train_size:]

    num_units = 128
    embedding_dim = 16
    num_products = len(product_mapping)
    target_size = len(target_indices)
    learning_rate = 0.001
    batch_size = 32
    epochs = 50

    logger.info("Building and training LSTM model")
    model, mae, rmse = train_and_evaluate_model(
        product_train, X_train, product_test, X_test, y_train, y_test,
        num_units, batch_size, epochs, learning_rate, seq_length, target_size, num_products, embedding_dim
    )

    logger.info(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    model_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()