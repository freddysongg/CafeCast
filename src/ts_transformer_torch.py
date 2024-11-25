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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

    Example:
        If `data` contains transaction quantities and revenues, and `product_ids` contains product mappings,
        this function generates sequences with a fixed length (`seq_length`) for model input.
    """
    X, y, products = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_indices])
        products.append(product_ids[i:i + seq_length])
    return np.array(products), np.array(X), np.array(y)

def create_gradient(rmse_change, overfit_threshold=0.01, underfit_threshold=0.05):
    """
    Creates a gradient to adjust hyperparameters dynamically based on model performance.

    Args:
        rmse_change (float): The change in RMSE between iterations (negative for improvement).
        overfit_threshold (float): Threshold for identifying potential overfitting.
        underfit_threshold (float): Threshold for identifying potential underfitting.

    Returns:
        dict: Gradient indicating 'increase' or 'decrease' for each hyperparameter.
    """
    gradient = {}

    if rmse_change > underfit_threshold:
        # Model underfits; increase capacity
        gradient = {
            'num_heads': 'increase',
            'num_layers': 'increase',
            'd_model': 'increase',
            'ff_dim': 'increase',
            'learning_rate': 'increase',
            'dropout_rate': 'decrease',  # Reduce regularization to avoid underfitting
        }
    elif rmse_change < -overfit_threshold:
        # Model overfits; decrease capacity or increase regularization
        gradient = {
            'num_heads': 'decrease',
            'num_layers': 'decrease',
            'd_model': 'decrease',
            'ff_dim': 'decrease',
            'learning_rate': 'decrease',
            'dropout_rate': 'increase',  # Add regularization to mitigate overfitting
        }
    else:
        gradient = {
            'num_heads': 'increase',
            'num_layers': 'decrease',
            'd_model': 'increase',
            'ff_dim': 'decrease',
            'learning_rate': 'decrease',
            'dropout_rate': 'decrease',
        }

    return gradient

class TransformerModel(nn.Module):
    """
    Transformer-based model for time-series forecasting with product embeddings.

    Attributes:
        embedding (nn.Embedding): Embedding layer for product IDs.
        projection (nn.Linear): Linear layer to project numerical inputs to the same dimension as embeddings.
        transformer_layers (nn.TransformerEncoder): Stacked Transformer encoder layers for processing sequences.
        global_pool (nn.AdaptiveAvgPool1d): Adaptive average pooling layer to reduce sequence dimension.
        dropout (nn.Dropout): Dropout layer for regularization.
        output_layer (nn.Linear): Linear output layer to predict target values.

    Methods:
        forward(numerical_input, product_input):
            Combines numerical features and product embeddings, passes them through the Transformer layers, and 
            outputs predictions.
    """
    def __init__(self, seq_length, num_features, num_heads, num_layers, d_model, ff_dim, target_size, num_products, dropout_rate):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_products, d_model)
        self.projection = nn.Linear(num_features, d_model)  
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_rate, batch_first=True),
            num_layers=num_layers
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(d_model, target_size)

    def forward(self, numerical_input, product_input):
        product_embedded = self.embedding(product_input) 
        numerical_projected = self.projection(numerical_input) 

        x = numerical_projected + product_embedded  

        x = self.transformer_layers(x) 

        x = x.permute(0, 2, 1) 
        x = self.global_pool(x).squeeze(-1) 
        x = self.dropout(x)
        return self.output_layer(x) 

def train_and_evaluate_model(inputs_train, inputs_test, y_train, y_test, num_heads, num_layers, d_model, ff_dim, learning_rate, dropout_rate, seq_length, target_size, num_products, device):
    """
    Trains and evaluates the Transformer model using PyTorch.

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
        num_products (int): Number of unique product IDs for embedding.
        device (torch.device): Device to run the model on ('cuda' for GPU, 'cpu' otherwise).

    Returns:
        TransformerModel: The trained Transformer model.
        float: Root Mean Squared Error (RMSE) on the test dataset.

    Notes:
        - The function uses PyTorch's `DataLoader` for efficient batching during training.
        - Regularization is applied using dropout layers in the model.
    """
    product_train, X_train = inputs_train
    product_test, X_test = inputs_test

    model = TransformerModel(seq_length, X_train.shape[2], num_heads, num_layers, d_model, ff_dim, target_size, num_products, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(product_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):  # 50 epochs
        for numerical_input, product_input, targets in train_loader:
            numerical_input, product_input, targets = numerical_input.to(device), product_input.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(numerical_input, product_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        product_test_tensor = torch.tensor(product_test, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        predictions = model(X_test_tensor, product_test_tensor).cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return model, rmse

def dynamic_param_tuning(best_params, gradient):
    """
    Dynamically adjusts hyperparameters based on a gradient or exploration strategy.

    Args:
        best_params (dict): Current best hyperparameters.
        gradient (dict): Dictionary indicating parameter adjustment directions (e.g., 'increase', 'decrease').

    Returns:
        dict: Updated hyperparameters.

    Example:
        If `gradient` is {'num_heads': 'increase'}, the function increases the number of attention heads.
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
        - If any required keys are missing, default values are added before saving.
        - The parameters are saved to `params/best_ts_transformer_params.json`.
    """
    params_path = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')

    required_keys = ['num_heads', 'num_layers', 'd_model', 'ff_dim', 'learning_rate', 'dropout_rate']
    for key in required_keys:
        if key not in best_params:
            logger.warning(f"Missing parameter '{key}' in best_params. Adding default value.")
            best_params.setdefault(key, 0.001 if key == 'learning_rate' else 0.1)

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
            return {**default_params, **params}

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
        5. Iteratively trains and evaluates the Transformer model with the current parameters using gradient-based tuning.
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
    max_iterations = 10

    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}: Testing parameters {best_params}")
        model, rmse, val_loss = train_and_evaluate_model(
            [product_train, X_train], [product_test, X_test],
            y_train, y_test, best_params['num_heads'], best_params['num_layers'],
            best_params['d_model'], best_params['ff_dim'], best_params['learning_rate'],
            best_params['dropout_rate'], seq_length, len(target_indices)
        )

        logger.info(f"Iteration {iteration + 1} Results - RMSE: {rmse:.2f}, Validation Loss: {val_loss:.4f}")

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

        gradient = {
            'learning_rate': 'decrease' if val_loss > 0.05 else 'increase',
            'num_heads': 'increase' if rmse > 0.1 else 'decrease',
            'num_layers': 'increase' if rmse > 0.1 else 'decrease',
            'ff_dim': 'increase' if val_loss > 0.05 else 'decrease',
            'dropout_rate': 'increase' if val_loss < 0.05 else 'decrease',
        }

        best_params = dynamic_param_tuning(best_params, gradient)
        logger.info(f"Updated parameters for next iteration: {best_params}")

    logger.info(f"Final RMSE after tuning: {best_rmse:.2f}")

if __name__ == "__main__":
    main()