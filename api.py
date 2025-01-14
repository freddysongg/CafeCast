from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import torch

app = FastAPI()

# Paths for models and scaler
SCALER_PATH = "models/scaler.pkl"
LSTM_MODEL_PATH = "models/scaler_lstm.pkl"
LSTM_BAYESIAN_MODEL_PATH = "models/scaler_lstm_bayesian.pkl"
TRANSFORMER_MODEL_PATH = "models/scaler_ts_transformer.pkl"

# Globals for models and scaler
scaler = None
lstm_model = None
lstm_bayesian_model = None
transformer_model = None

@app.on_event("startup")
def load_models_and_scaler():
    global scaler, lstm_model, lstm_bayesian_model, transformer_model

    # Load scaler
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = joblib.load(f)
        print("Scaler loaded successfully.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Scaler file not found.")

    # Load LSTM model
    try:
        with open(LSTM_MODEL_PATH, "rb") as f:
            lstm_model = joblib.load(f)
        print("LSTM model loaded successfully.")
    except Exception as e:
        print(f"Error loading LSTM model: {e}")

    # Load Bayesian LSTM model
    try:
        with open(LSTM_BAYESIAN_MODEL_PATH, "rb") as f:
            lstm_bayesian_model = joblib.load(f)
        print("Bayesian LSTM model loaded successfully.")
    except Exception as e:
        print(f"Error loading Bayesian LSTM model: {e}")

    # Load Transformer model
    try:
        with open(TRANSFORMER_MODEL_PATH, "rb") as f:
            transformer_model = torch.load(f, map_location="cpu")
            print(transformer_model.state_dict().keys())
            transformer_model.eval()
        print("Transformer model loaded successfully.")
    except Exception as e:
        print(f"Error loading Transformer model: {e}")

# Request model
class PredictionRequest(BaseModel):
    model: str  # 'lstm', 'lstm_bayesian', 'transformer'
    input_values: list  # Raw input values

@app.post("/predict")
def predict(data: PredictionRequest):
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler not loaded.")

    # Preprocess input
    try:
        input_array = np.array(data.input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Predict based on model
    if data.model == "lstm":
        if lstm_model:
            prediction = lstm_model.predict(scaled_input)
        else:
            raise HTTPException(status_code=500, detail="LSTM model not loaded.")
    elif data.model == "lstm_bayesian":
        if lstm_bayesian_model:
            prediction = lstm_bayesian_model.predict(scaled_input)
        else:
            raise HTTPException(status_code=500, detail="Bayesian LSTM model not loaded.")
    elif data.model == "transformer":
        if transformer_model:
            with torch.no_grad():
                tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
                prediction = transformer_model(tensor_input).numpy()
        else:
            raise HTTPException(status_code=500, detail="Transformer model not loaded.")
    else:
        raise HTTPException(status_code=400, detail="Invalid model type.")

    return {"model": data.model, "prediction": prediction.tolist()}
