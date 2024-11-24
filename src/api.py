from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import torch
from torch import load
import joblib
from statsmodels.tsa.arima.model import ARIMAResults
import os

from models.time_series_transformer import TimeSeriesTransformer

app = FastAPI()

# Model and Scaler Paths
LSTM_MODEL_PATH = "models/best_lstm_model.keras"
LSTM_BAYESIAN_MODEL_PATH = "models/best_lstm_bayesian_model.keras"
TRANSFORMER_MODEL_PATH = "models/best_ts_transformer_model.pt"
ARIMA_MODEL_PATH = "models/arima_model.pkl"
LSTM_SCALER_PATH = "models/scaler_lstm.pkl"
LSTM_BAYESIAN_SCALER_PATH = "models/scaler_lstm_bayesian.pkl"
TRANSFORMER_SCALER_PATH = "models/scaler_ts_transformer.pkl"

# Models and Scalers
lstm_model = None
lstm_bayesian_model = None
transformer_model = None
arima_model = None
lstm_scaler = None
lstm_bayesian_scaler = None
transformer_scaler = None

@app.on_event("startup")
def load_models_and_scalers():
    global lstm_model, lstm_bayesian_model, transformer_model, arima_model
    global lstm_scaler, lstm_bayesian_scaler, transformer_scaler

    # Load LSTM model
    if os.path.exists(LSTM_MODEL_PATH):
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        print("LSTM model loaded successfully.")

    # Load LSTM Bayesian model
    if os.path.exists(LSTM_BAYESIAN_MODEL_PATH):
        lstm_bayesian_model = tf.keras.models.load_model(LSTM_BAYESIAN_MODEL_PATH)
        print("LSTM Bayesian model loaded successfully.")

    # Load Transformer model
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        transformer_model = TimeSeriesTransformer(
            input_size=64,
            num_layers=2,
            num_heads=16,
            d_model=64,
            dim_feedforward=192
        )
        state_dict = torch.load(TRANSFORMER_MODEL_PATH, map_location="cpu")
        transformer_model.load_state_dict(state_dict, strict=False)
        transformer_model.eval()
        print("Transformer model loaded successfully.")

    # Load ARIMA model
    # if os.path.exists(ARIMA_MODEL_PATH):
    #     with open(ARIMA_MODEL_PATH, "rb") as f:
    #         arima_model = ARIMAResults.load(f)
    #     print("ARIMA model loaded successfully.")

    # Load LSTM scaler
    if os.path.exists(LSTM_SCALER_PATH):
        lstm_scaler = joblib.load(LSTM_SCALER_PATH)
        print("LSTM scaler loaded successfully.")

    # Load LSTM Bayesian scaler
    if os.path.exists(LSTM_BAYESIAN_SCALER_PATH):
        lstm_bayesian_scaler = joblib.load(LSTM_BAYESIAN_SCALER_PATH)
        print("LSTM Bayesian scaler loaded successfully.")

    # Load Transformer scaler
    if os.path.exists(TRANSFORMER_SCALER_PATH):
        transformer_scaler = joblib.load(TRANSFORMER_SCALER_PATH)
        print("Transformer scaler loaded successfully.")

    print("All models and scalers loaded successfully.")

# Input Data Schema
class InputData(BaseModel):
    data: list

@app.post("/predict/lstm")
def predict_lstm(input_data: InputData):
    if lstm_model is None or lstm_scaler is None:
        raise HTTPException(status_code=500, detail="LSTM model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = lstm_scaler.transform(input_array)

        # Create sequences
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        predictions = lstm_model.predict(sequences)
        predictions = lstm_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lstm-bayesian")
def predict_lstm_bayesian(input_data: InputData):
    if lstm_bayesian_model is None or lstm_bayesian_scaler is None:
        raise HTTPException(status_code=500, detail="LSTM Bayesian model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = lstm_bayesian_scaler.transform(input_array)

        # Create sequences
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        predictions = lstm_bayesian_model.predict(sequences)
        predictions = lstm_bayesian_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/transformer")
def predict_transformer(input_data: InputData):
    if transformer_model is None or transformer_scaler is None:
        raise HTTPException(status_code=500, detail="Transformer model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = transformer_scaler.transform(input_array)

        # Create sequences
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        sequences = torch.tensor(sequences, dtype=torch.float32)

        # Get predictions
        transformer_model.eval()
        with torch.no_grad():
            predictions = transformer_model(sequences).numpy()
        predictions = transformer_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Model inference API is running."}
