from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import torch
from torch import load, nn
import torch.nn as nn
import joblib
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.preprocessing import MinMaxScaler
import json
import os

from models.time_series_transformer import TimeSeriesTransformer

app = FastAPI()

# Load Models and Parameters
LSTM_MODEL_PATH = "models/best_lstm_model.keras"
LSTM_BAYESIAN_MODEL_PATH = "models/best_lstm_bayesian_model.keras"
TRANSFORMER_MODEL_PATH = "models/best_ts_transformer_model.pt"
ARIMA_MODEL_PATH = "models/arima_model.pkl"
PARAMS_DIR = "params/"
SCALER_PATH = "models/scaler.pkl"

# LSTM Model
lstm_model = None

# LSTM Bayesian Model
lstm_bayesian_model = None

# Transformer Model
transformer_model = None

# ARIMA Model
arima_model = None

# Scaler
scaler = None

@app.on_event("startup")
def load_models():
    global lstm_model, lstm_bayesian_model, transformer_model, arima_model, scaler

    # Load LSTM model
    if os.path.exists(LSTM_MODEL_PATH):
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        
    # Load LSTM Bayesian model
    if os.path.exists(LSTM_BAYESIAN_MODEL_PATH):
        lstm_bayesian_model = tf.keras.models.load_model(LSTM_BAYESIAN_MODEL_PATH)

    # Load Transformer model
    if os.path.exists(TRANSFORMER_MODEL_PATH):
        transformer_model = TimeSeriesTransformer(
            input_size=64, 
            num_layers=2, 
            num_heads=16, 
            d_model=64, 
            dim_feedforward=192
        )
        state_dict = torch.load(TRANSFORMER_MODEL_PATH, map_location='cpu')
        transformer_model.load_state_dict(state_dict, strict=False)
        
        transformer_model.eval()

    # Load ARIMA model
    # if os.path.exists(ARIMA_MODEL_PATH):
    #     with open(ARIMA_MODEL_PATH, "rb") as f:
    #         arima_model = ARIMAResults.load(f)

    # Load Scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = joblib.load(f)

    print("Models and scaler loaded successfully.")

# Input Data Schema
class InputData(BaseModel):
    data: list

@app.post("/predict/lstm")
def predict_lstm(input_data: InputData):
    if lstm_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="LSTM model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Create sequences (assuming SEQ_LENGTH = 10 as used in training)
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        predictions = lstm_model.predict(sequences)
        predictions = scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/lstm-bayesian")
def predict_lstm_bayesian(input_data: InputData):
    if lstm_bayesian_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="LSTM Bayesian model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Create sequences (assuming SEQ_LENGTH = 10 as used in training)
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        predictions = lstm_bayesian_model.predict(sequences)
        predictions = scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/transformer")
def predict_transformer(input_data: InputData):
    if transformer_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Transformer model or scaler not loaded.")

    try:
        # Scale input data
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)

        # Create sequences (SEQ_LENGTH = 10)
        SEQ_LENGTH = 10
        if len(scaled_input) < SEQ_LENGTH:
            raise ValueError("Input data must have at least 10 data points.")

        sequences = np.array([scaled_input[i:i + SEQ_LENGTH] for i in range(len(scaled_input) - SEQ_LENGTH + 1)])
        sequences = torch.tensor(sequences, dtype=torch.float32)

        # Get predictions
        transformer_model.eval()
        with torch.no_grad():
            predictions = transformer_model(sequences).numpy()
        predictions = scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/arima")
def predict_arima(input_data: InputData):
    if arima_model is None:
        raise HTTPException(status_code=500, detail="ARIMA model not loaded.")

    try:
        # Use ARIMA to forecast based on the length of input data
        forecast_steps = len(input_data.data)
        forecast = arima_model.forecast(steps=forecast_steps)

        return {"predictions": forecast.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Model inference API is running."}
