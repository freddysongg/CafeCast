import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import sys
import os
import logging
import tensorflow as tf

# Configure logging for the main script
logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Paths to parameter files
PARAMS_DIR = 'params/'
LSTM_PARAMS_FILE = os.path.join(PARAMS_DIR, 'best_lstm_params.json')
TS_TRANSFORMER_PARAMS_FILE = os.path.join(PARAMS_DIR, 'best_ts_transformer_params.json')

# Ensure the params directory exists
os.makedirs(PARAMS_DIR, exist_ok=True)

def clear_params(model_type):
    """
    Deletes the parameter file for the specified model type if it exists.

    Args:
        model_type (str): The type of model whose parameter file should be cleared.
                          Acceptable values:
                          - 'lstm': Clears LSTM parameter file.
                          - 'ts_transformer': Clears Transformer parameter file.

    Behavior:
        - Deletes the relevant parameter file (`best_lstm_params.json` or `best_ts_transformer_params.json`).
        - Logs a message indicating the action taken.
        - If the parameter file does not exist or is already cleared, logs that information.

    Examples:
        clear_params('lstm')  # Deletes the LSTM parameter file if it exists.
    """
    if model_type == 'lstm' and os.path.exists(LSTM_PARAMS_FILE):
        os.remove(LSTM_PARAMS_FILE)
        logger.info(f"Cleared LSTM parameter file: {LSTM_PARAMS_FILE}")
    elif model_type == 'ts_transformer' and os.path.exists(TS_TRANSFORMER_PARAMS_FILE):
        os.remove(TS_TRANSFORMER_PARAMS_FILE)
        logger.info(f"Cleared Transformer parameter file: {TS_TRANSFORMER_PARAMS_FILE}")
    else:
        logger.info(f"No parameter file found for {model_type} or file already cleared.")

def main():
    """
    Main function to handle menu-based execution of different models and utilities.

    Behavior:
        - Detects if the operating system is Windows.
        - Adjusts the model file names to use the `_torch.py` versions for Windows.
        - Logs which implementation is running.
        - Displays a menu of options to the user:
          1. Run LSTM Model
          2. Run Time Series Transformer Model
          3. Run Bayesian LSTM Optimization
          4. Clear LSTM Model Parameters
          5. Clear Transformer Model Parameters
          6. Run ARIMA Model
        - Executes the corresponding functionality based on the user's input.
    """
    is_windows = os.name == 'nt'  # Check if running on Windows

    lstm_module = 'lstm_torch' if is_windows else 'lstm'
    ts_transformer_module = 'ts_transformer_torch' if is_windows else 'ts_transformer'
    lstm_bayesian_module = 'lstm_bayesian_torch' if is_windows else 'lstm_bayesian'

    logger.info("Choose an option:")
    logger.info("1. Run LSTM Model")
    logger.info("2. Run Time Series Transformer Model")
    logger.info("3. Run Bayesian LSTM Optimization")
    logger.info("4. Clear LSTM Model Parameters")
    logger.info("5. Clear Transformer Model Parameters")
    logger.info("6. Run ARIMA Model")
    choice = input("Enter the number of your choice: ")

    try:
        if choice == '1':
            logger.info(f"Running LSTM Model using {'Torch' if is_windows else 'TensorFlow'} implementation.")
            module = __import__(lstm_module)
            module.main()
        elif choice == '2':
            logger.info(f"Running Time Series Transformer Model using {'Torch' if is_windows else 'TensorFlow'} implementation.")
            module = __import__(ts_transformer_module)
            module.main()
        elif choice == '3':
            logger.info(f"Running Bayesian LSTM Optimization using {'Torch' if is_windows else 'TensorFlow'} implementation.")
            module = __import__(lstm_bayesian_module)
            module.main()
        elif choice == '4':
            logger.info("Clearing LSTM Model Parameters.")
            clear_params('lstm')
        elif choice == '5':
            logger.info("Clearing Transformer Model Parameters.")
            clear_params('ts_transformer')
        elif choice == '6':
            logger.info("Running ARIMA Model.")
            import arima
            arima.main()
        else:
            logger.info("Invalid choice. Please select a valid option.")
    except ImportError as e:
        logger.error(f"Error importing module: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
