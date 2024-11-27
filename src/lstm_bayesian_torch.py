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
from functools import partial
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from preprocess_data import process_data

LOG_DIR = 'logs/'
MODEL_DIR = 'models/'
PARAMS_DIR = 'params/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

cache = {}

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

class LSTMModel(nn.Module):
    """
    Defines a PyTorch-based LSTM model with product embeddings and a fully connected output layer.

    Args:
        input_size (int): Number of numerical input features per time step.
        num_units (int): Number of hidden units in the LSTM layer.
        output_size (int): Number of output targets (e.g., predictions per time step).
        num_products (int): Number of unique product IDs for embedding.
        embedding_dim (int): Dimension of the product embedding.

    Methods:
        forward(numerical_input, product_input):
            Combines numerical features with product embeddings, processes them through LSTM layers, 
            and returns predictions.

            Args:
                numerical_input (torch.Tensor): Numerical features of shape (batch_size, seq_length, input_size).
                product_input (torch.Tensor): Product ID features of shape (batch_size, seq_length).
            
            Returns:
                torch.Tensor: Predictions of shape (batch_size, output_size).
    """
    def __init__(self, input_size, num_units, output_size, num_products, embedding_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_products, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, num_units, batch_first=True)
        self.fc = nn.Linear(num_units, output_size)

    def forward(self, numerical_input, product_input):
        embedded_products = self.embedding(product_input)
        combined_input = torch.cat((numerical_input, embedded_products), dim=2)
        lstm_out, _ = self.lstm(combined_input)
        output = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return output

def train_and_evaluate_model(product_train, X_train, product_test, X_test, y_train, y_test, num_units, batch_size, epochs, learning_rate, seq_length, target_size, num_products, embedding_dim, device):
    """
    Trains and evaluates a PyTorch LSTM model with product embeddings.

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
        target_size (int): Number of target outputs (e.g., predictions per time step).
        num_products (int): Number of unique product IDs for embedding.
        embedding_dim (int): Dimension of the product embedding.
        device (torch.device): Device to run the model on ('cuda' for GPU, 'cpu' otherwise).

    Returns:
        LSTMModel: The trained PyTorch LSTM model.
        float: Root Mean Squared Error (RMSE) on the test dataset.
    """
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, int(num_units), target_size, num_products, embedding_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(product_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)

    model.train()
    for epoch in range(int(epochs)):
        for numerical_input, product_input, target in train_loader:
            numerical_input, product_input, target = numerical_input.to(device), product_input.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(numerical_input, product_input)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        product_test_tensor = torch.tensor(product_test, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        predictions = model(X_test_tensor, product_test_tensor).cpu().numpy()
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

def objective_function(num_units, batch_size, epochs, learning_rate, seq_length, target_indices, product_train, X_train, product_test, X_test, y_train, y_test, num_products, embedding_dim):
    """
    Objective function for Bayesian Optimization. Trains the LSTM model and evaluates its performance 
    using the provided hyperparameters.

    Args:
        num_units (float): Number of LSTM units (treated as a float by Bayesian Optimization).
        batch_size (float): Batch size for training (treated as a float by Bayesian Optimization).
        epochs (float): Number of epochs for training (treated as a float by Bayesian Optimization).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Negative RMSE (Root Mean Squared Error). The negative value is used because Bayesian 
               Optimization maximizes the objective function, and we aim to minimize RMSE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, rmse = train_and_evaluate_model(
        product_train, X_train, product_test, X_test, y_train, y_test,
        num_units=num_units, batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, seq_length=seq_length, target_size=len(target_indices),
        num_products=num_products, embedding_dim=embedding_dim, device=device
    )
    logger.info(f"Tested params: num_units={num_units}, batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}, RMSE={rmse:.2f}")
    return -rmse

def objective_function_with_cache(num_units, batch_size, epochs, learning_rate, seq_length, target_indices, product_train, X_train, product_test, X_test, y_train, y_test, num_products, embedding_dim):
    """
    Objective function with caching to avoid redundant computations.

    Args:
        num_units (float): Number of LSTM units.
        batch_size (float): Batch size for training.
        epochs (float): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Negative RMSE (for Bayesian Optimization).
    """
    key = (
        int(num_units), int(batch_size), int(epochs), 
        float(learning_rate), int(seq_length), 
        tuple(target_indices), num_products, embedding_dim
    )

    if key in cache:
        logger.info(f"Using cached result for params: {key}")
        return cache[key]

    rmse = objective_function(
        num_units, batch_size, epochs, learning_rate, seq_length, target_indices,
        product_train, X_train, product_test, X_test, y_train, y_test, num_products, embedding_dim
    )
    cache[key] = rmse
    return rmse

def parallel_evaluate(params_list, seq_length, target_indices, product_train, X_train, product_test, X_test, y_train, y_test, num_products, embedding_dim):
    """
    Evaluates multiple parameter sets in parallel.

    Args:
        params_list (list): List of parameter dictionaries to evaluate.

    Returns:
        list: List of evaluation results corresponding to the parameter sets.
    """
    return Parallel(n_jobs=2)(
        delayed(objective_function_with_cache)(
            params['num_units'],
            params['batch_size'],
            params['epochs'],
            params['learning_rate'],
            seq_length,
            target_indices,
            product_train,
            X_train,
            product_test,
            X_test,
            y_train,
            y_test,
            num_products,
            embedding_dim
        )
        for params in params_list
    )

def generate_unique_params(bounds, n_params, seen_params):
    """
    Generates a unique set of parameter suggestions for Bayesian Optimization.
    
    Args:
        bounds (dict): Parameter bounds for the optimization.
        n_params (int): Number of unique parameter sets to generate.
        seen_params (set): A set of previously seen parameter sets.
        
    Returns:
        list: A list of unique parameter dictionaries.
    """
    unique_params = []
    while len(unique_params) < n_params:
        new_params = {
            'num_units': np.random.uniform(*bounds['num_units']),
            'batch_size': np.random.uniform(*bounds['batch_size']),
            'epochs': np.random.uniform(*bounds['epochs']),
            'learning_rate': np.random.uniform(*bounds['learning_rate']),
        }
        params_tuple = tuple(new_params.items())
        if params_tuple not in seen_params:
            unique_params.append(new_params)
            seen_params.add(params_tuple)
    return unique_params

def main():
    """
    Main function for running Bayesian Optimization on a PyTorch LSTM model.

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
        - Assumes global variables for processed data (`X_train`, `X_test`, `y_train`, etc.).
        - Saves the model to `models/best_lstm_model.pth`.
        - Saves the best hyperparameters to `params/best_lstm_bayesian_params.json`.
        - Implements caching to reuse results for previously tested parameter sets.
        - Parallelizes the evaluations using joblib for faster computation.
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

    logger.info("Running LSTM Bayesian Optimization using PyTorch")

    global seq_length, target_indices, X_train, X_test, y_train, y_test, product_train, product_test, num_products, embedding_dim
    seq_length = 10
    target_indices = [0, 1]
    product_sequences, X, y = create_sequences(scaled_data, product_ids, seq_length, target_indices)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    product_train, product_test = product_sequences[:train_size], product_sequences[train_size:]

    num_products = len(product_mapping)
    embedding_dim = 8

    bounds = {
        'num_units': (64, 128),
        'batch_size': (16, 32),
        'epochs': (10, 50),
        'learning_rate': (0.0001, 0.01),
    }

    bound_objective = partial(
        objective_function_with_cache,
        seq_length=seq_length,
        target_indices=target_indices,
        product_train=product_train,
        X_train=X_train,
        product_test=product_test,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_products=num_products,
        embedding_dim=embedding_dim
    )

    optimizer = BayesianOptimization(
        f=bound_objective,
        pbounds=bounds,
        verbose=2,
        random_state=42,
    )

    init_points = 5
    optimizer.maximize(init_points=init_points, n_iter=0)
    torch.cuda.empty_cache()
    seen_params = set()

    # Run Bayesian Optimization iterations with batch parallelization
    n_iter = 10
    for _ in range(n_iter):
        logger.info("Generating batch of suggestions.")
        suggested_params = generate_unique_params(bounds, 5, seen_params)
        logger.info(f"Evaluating {len(suggested_params)} parameter sets in parallel.")
        
        # Evaluate in parallel
        results = parallel_evaluate(
            suggested_params,
            seq_length,
            target_indices,
            product_train,
            X_train,
            product_test,
            X_test,
            y_train,
            y_test,
            num_products,
            embedding_dim
        )
        for params, result in zip(suggested_params, results):
            optimizer.register(params=params, target=result)

    best_params = optimizer.max['params']
    logger.info(f"Best parameters found: {best_params}")
    save_best_params(best_params)

if __name__ == "__main__":
    main()