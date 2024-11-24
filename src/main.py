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
    if model_type == 'lstm' and os.path.exists(LSTM_PARAMS_FILE):
        os.remove(LSTM_PARAMS_FILE)
        logger.info(f"Cleared LSTM parameter file: {LSTM_PARAMS_FILE}")
    elif model_type == 'ts_transformer' and os.path.exists(TS_TRANSFORMER_PARAMS_FILE):
        os.remove(TS_TRANSFORMER_PARAMS_FILE)
        logger.info(f"Cleared Transformer parameter file: {TS_TRANSFORMER_PARAMS_FILE}")
    else:
        logger.info(f"No parameter file found for {model_type} or file already cleared.")

def main():
    logger.info("Choose an option:")
    logger.info("1. Run LSTM Model")
    logger.info("2. Run Time Series Transformer Model")
    logger.info("3. Run Bayesian LSTM Optimization")
    logger.info("4. Clear LSTM Model Parameters")
    logger.info("5. Clear Transformer Model Parameters")
    logger.info("6. Run ARIMA Model")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        from lstm import main as run_lstm
        run_lstm()
    elif choice == '2':
        from ts_transformer import main as run_ts_transformer
        run_ts_transformer()
    elif choice == '3':
        from lstm_bayesian import main as run_lstm_bayesian
        run_lstm_bayesian()
    elif choice == '4':
        clear_params('lstm')
    elif choice == '5':
        clear_params('ts_transformer')
    elif choice == '6':
        import arima
        arima.main()
    else:
        logger.info("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
