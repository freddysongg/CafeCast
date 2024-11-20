import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import os
import glob
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data.modify_dataset import prepare_data

LOG_DIR = 'logs/'
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"lstm_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cleanup_old_logs(directory, prefix="lstm_log_", max_logs=5):
    log_files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.log")), key=os.path.getmtime)
    if len(log_files) > max_logs:
        for log_file in log_files[:-max_logs]:
            os.remove(log_file)
            logging.info(f"Deleted old log file: {log_file}")

def main():
    # Log the start of the script
    logging.info("Starting LSTM model script")

    # Cleanup old logs
    cleanup_old_logs(LOG_DIR)

    # Load and prepare data
    logging.info("Loading and preparing data")
    data = prepare_data('data/cafecast_data.xlsx')
    daily_data = data.resample('D')['transaction_qty'].sum()

    # Log data overview
    logging.info(f"Data overview: {daily_data.describe()}")

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))

    # Create sequences for LSTM input
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    SEQ_LENGTH = 10  # Adjust as needed
    logging.info(f"Sequence length set to: {SEQ_LENGTH}")
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Training data size: {len(X_train)}, Testing data size: {len(X_test)}")

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Log model configuration
    logging.info("Model configuration:")
    model.summary(print_fn=lambda x: logging.info(x))

    # Train the model
    logging.info("Starting model training")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Log training history
    logging.info("Training completed. Logging training history:")
    for epoch, loss, val_loss in zip(
        range(1, len(history.history['loss']) + 1),
        history.history['loss'],
        history.history['val_loss']
    ):
        logging.info(f"Epoch {epoch}: loss = {loss:.4f}, val_loss = {val_loss:.4f}")

    # Make predictions
    logging.info("Making predictions")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted values
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_actual, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red')
    plt.title('Actual vs Predicted Transaction Quantities (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Transaction Quantity')
    plt.legend()
    plt.show()

    # Evaluate the model
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    logging.info(f"Model evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

if __name__ == "__main__":
    main()