import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import logging
import sys
import json
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from bayes_opt import BayesianOptimization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from preprocess_data import process_data

LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

evaluation_cache = defaultdict(lambda: None)

log_filename = os.path.join(LOG_DIR, f"lstm_bayesian_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
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
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return model, rmse

def save_best_params(best_params):
    """
    Saves the best hyperparameters dynamically to a JSON file.

    Args:
        best_params (dict): Dictionary containing the best hyperparameters. 
                            Keys must include 'num_units', 'batch_size', 'epochs', and 'learning_rate'.

    Notes:
        If any required keys are missing, default values are added before saving.
        The parameters are saved as a JSON file (`params/best_lstm_bayesian_params.json`).
    """
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_bayesian_params.json')

    required_keys = ['num_units', 'batch_size', 'epochs', 'learning_rate']
    for key in required_keys:
        if key not in best_params:
            logger.warning(f"Missing parameter '{key}' in best_params before saving. Adding default value.")
            if key == 'learning_rate':
                best_params[key] = 0.001
            elif key == 'num_units':
                best_params[key] = 128
            elif key == 'batch_size':
                best_params[key] = 32
            elif key == 'epochs':
                best_params[key] = 50

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
        - num_units: 128
        - batch_size: 32
        - epochs: 50
        - learning_rate: 0.001
    """
    params_path = os.path.join(PARAMS_DIR, 'best_lstm_bayesian_params.json')

    default_params = {'num_units': 128, 'batch_size': 32, 'epochs': 50, 'learning_rate': 0.001}

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

def objective_function(num_units, batch_size, epochs, learning_rate):
    """
    Objective function for Bayesian Optimization. Trains the LSTM model and evaluates its performance 
    using the provided hyperparameters. Uses caching to avoid redundant evaluations.

    Args:
        num_units (float): Number of LSTM units (treated as a float by Bayesian Optimization).
        batch_size (float): Batch size for training (treated as a float by Bayesian Optimization).
        epochs (float): Number of epochs for training (treated as a float by Bayesian Optimization).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Negative RMSE (Root Mean Squared Error). The negative value is used because Bayesian 
               Optimization maximizes the objective function, and we aim to minimize RMSE.
    """
    params_key = (int(num_units), int(batch_size), int(epochs), round(learning_rate, 6))
    
    if params_key in evaluation_cache:
        logger.info(f"Using cached result for parameters: {params_key}")
        return evaluation_cache[params_key]

    model, rmse = train_and_evaluate_model(
        product_train, X_train, product_test, X_test, y_train, y_test,
        num_units=num_units, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, seq_length=seq_length, target_size=len(target_indices),
        num_products=num_products, embedding_dim=embedding_dim
    )
    logger.info(f"Tested params: num_units={num_units}, batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}, RMSE={rmse:.2f}")
    
    # Cache the result
    evaluation_cache[params_key] = -rmse  # Negative because Bayesian Optimization maximizes
    return -rmse

def parallel_objective_function(params_list):
    """
    Wrapper to run multiple evaluations in parallel using `joblib`.

    Args:
        params_list (list): A list of parameter dictionaries to evaluate in parallel.

    Returns:
        list: A list of results for the objective function for each parameter set.
    """
    results = Parallel(n_jobs=-1)(
        delayed(objective_function)(
            num_units=p['num_units'],
            batch_size=p['batch_size'],
            epochs=p['epochs'],
            learning_rate=p['learning_rate']
        ) for p in params_list
    )
    return results

def main():
    """
    Main function for running Bayesian Optimization on a TensorFlow LSTM model.

    Steps:
        1. Loads and preprocesses the dataset.
        2. Maps product IDs to integers for embedding.
        3. Configures the Bayesian Optimization bounds for hyperparameters:
           - num_units: Number of LSTM hidden units.
           - batch_size: Batch size for training.
           - epochs: Number of epochs for training.
           - learning_rate: Learning rate for the optimizer.
        4. Runs the Bayesian Optimization process to find the best hyperparameters.
        5. Logs and saves the best parameters found by Bayesian Optimization.
        6. Saves the trained model to disk.

    Notes:
        - Uses parallel evaluations for Bayesian Optimization.
        - Saves the model to `models/best_lstm_model.h5`.
        - Saves the best hyperparameters to `params/best_lstm_bayesian_params.json`.
    """
    logger.info("Loading and preprocessing data")
    processed_file = 'data/processed_coffee_shop_data.csv'
    df = process_data(processed_file, 'data/lstm_bayesian_output.csv')

    unique_products = df['product_id'].unique()
    product_mapping = {product: idx for idx, product in enumerate(unique_products)}
    df['product_id'] = df['product_id'].map(product_mapping)

    features = ['transaction_qty', 'revenue']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    product_ids = df['product_id'].values

    scaler_path = os.path.join(MODEL_DIR, 'scaler_lstm_bayesian.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    global seq_length, target_indices, X_train, X_test, y_train, y_test, product_train, product_test, num_products, embedding_dim
    seq_length = 10
    target_indices = [0, 1]
    product_sequences, X, y = create_sequences(scaled_data, product_ids, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    product_train, product_test = product_sequences[:train_size], product_sequences[train_size:]

    num_products = len(product_mapping)
    embedding_dim = 16

    bounds = {
        'num_units': (64, 256),
        'batch_size': (16, 64),
        'epochs': (25, 100),
        'learning_rate': (0.0001, 0.01),
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=bounds,
        verbose=2,
        random_state=42,
    )

    init_points = 3
    n_iter = 10

    # Evaluate initial points in parallel
    initial_params = [
        {k: np.random.uniform(bounds[k][0], bounds[k][1]) for k in bounds.keys()}
        for _ in range(init_points)
    ]
    initial_results = parallel_objective_function(initial_params)

    for params, result in zip(initial_params, initial_results):
        optimizer.register(params, result)

    # Run Bayesian Optimization iterations in parallel
    for _ in range(n_iter):
        next_points = [optimizer.suggest(optimizer.space) for _ in range(-1)]
        next_results = parallel_objective_function(next_points)
        for params, result in zip(next_points, next_results):
            optimizer.register(params, result)

    best_params = optimizer.max['params']
    logger.info(f"Best parameters found: {best_params}")
    save_best_params(best_params)


if __name__ == "__main__":
    main()