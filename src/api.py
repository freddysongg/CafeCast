from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import torch
import joblib
import requests
import os

from lstm_bayesian_torch import create_sequences as lstm_create_sequences
from lstm_torch import create_sequences as lstm_pure_create_sequences
from ts_transformer_torch import create_sequences as transformer_create_sequences

app = FastAPI()

LSTM_MODEL_PATH = "models/best_lstm_model.keras"
LSTM_BAYESIAN_MODEL_PATH = "models/best_lstm_bayesian_model.keras"
TRANSFORMER_MODEL_PATH = "models/best_ts_transformer_model.pt"
ARIMA_MODEL_PATH = "models/arima_model.pkl"
LSTM_SCALER_PATH = "models/scaler_lstm.pkl"
LSTM_BAYESIAN_SCALER_PATH = "models/scaler_lstm_bayesian.pkl"
TRANSFORMER_SCALER_PATH = "models/scaler_ts_transformer.pkl"

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

    if os.path.exists(LSTM_MODEL_PATH):
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        print("LSTM model loaded successfully.")

    if os.path.exists(LSTM_BAYESIAN_MODEL_PATH):
        lstm_bayesian_model = tf.keras.models.load_model(LSTM_BAYESIAN_MODEL_PATH)
        print("LSTM Bayesian model loaded successfully.")

    if os.path.exists(TRANSFORMER_MODEL_PATH):
        transformer_model = torch.load(TRANSFORMER_MODEL_PATH, map_location="cpu")
        transformer_model.eval()
        print("Transformer model loaded successfully.")

    if os.path.exists(ARIMA_MODEL_PATH):
        with open(ARIMA_MODEL_PATH, "rb") as f:
            arima_model = joblib.load(f)
        print("ARIMA model loaded successfully.")

    lstm_scaler = joblib.load(LSTM_SCALER_PATH) if os.path.exists(LSTM_SCALER_PATH) else None
    lstm_bayesian_scaler = joblib.load(LSTM_BAYESIAN_SCALER_PATH) if os.path.exists(LSTM_BAYESIAN_SCALER_PATH) else None
    transformer_scaler = joblib.load(TRANSFORMER_SCALER_PATH) if os.path.exists(TRANSFORMER_SCALER_PATH) else None
    print("Scalers loaded successfully.")

class InputData(BaseModel):
    data: list

class TextInput(BaseModel):
    text: str

def preprocess_with_gemini(user_input: str):
    """
    Use the Gemini API to preprocess text input into structured data.
    """
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        raise ValueError("Gemini API key not set.")

    gemini_endpoint = "https://gemini.googleapis.com/v1beta1/text:analyze"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {gemini_api_key}"}
    payload = {"document": {"type": "PLAIN_TEXT", "content": user_input}, "encodingType": "UTF8"}

    response = requests.post(gemini_endpoint, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API request failed: {response.text}")

    return extract_relevant_features(response.json())

def extract_relevant_features(api_response):
    """
    Extract structured features from the Gemini API response.
    """
    entities = api_response.get("entities", [])
    numerical_data = [float(entity["value"]) for entity in entities if "value" in entity]
    product_ids = [entity.get("type", "unknown") for entity in entities]
    return np.array(numerical_data).reshape(-1, 1), product_ids

def convert_to_model_input(processed_data, product_ids, model_type, seq_length=10):
    """
    Convert structured data into model-ready format.
    """
    if model_type == "LSTM":
        return lstm_pure_create_sequences(processed_data, product_ids, seq_length, [0])
    elif model_type == "LSTM-Bayesian":
        return lstm_create_sequences(processed_data, product_ids, seq_length, [0])
    elif model_type == "Transformer":
        return transformer_create_sequences(processed_data, product_ids, seq_length, [0])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

@app.post("/predict/lstm")
def predict_lstm(input_data: InputData):
    """
    Predict using the LSTM model.
    """
    if lstm_model is None or lstm_scaler is None:
        raise HTTPException(status_code=500, detail="LSTM model or scaler not loaded.")
    try:
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = lstm_scaler.transform(input_array)

        seq_length = 10
        sequences = [scaled_input[i:i + seq_length] for i in range(len(scaled_input) - seq_length + 1)]
        predictions = lstm_model.predict(np.array(sequences))
        predictions = lstm_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lstm-bayesian")
def predict_lstm_bayesian(input_data: InputData):
    """
    Predict using the LSTM Bayesian model.
    """
    if lstm_bayesian_model is None or lstm_bayesian_scaler is None:
        raise HTTPException(status_code=500, detail="LSTM Bayesian model or scaler not loaded.")
    try:
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = lstm_bayesian_scaler.transform(input_array)

        seq_length = 10
        sequences = [scaled_input[i:i + seq_length] for i in range(len(scaled_input) - seq_length + 1)]
        predictions = lstm_bayesian_model.predict(np.array(sequences))
        predictions = lstm_bayesian_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/transformer")
def predict_transformer(input_data: InputData):
    """
    Predict using the Transformer model.
    """
    if transformer_model is None or transformer_scaler is None:
        raise HTTPException(status_code=500, detail="Transformer model or scaler not loaded.")
    try:
        input_array = np.array(input_data.data).reshape(-1, 1)
        scaled_input = transformer_scaler.transform(input_array)

        seq_length = 10
        sequences = [scaled_input[i:i + seq_length] for i in range(len(scaled_input) - seq_length + 1)]
        sequences = torch.tensor(sequences, dtype=torch.float32)

        with torch.no_grad():
            predictions = transformer_model(sequences).numpy()
        predictions = transformer_scaler.inverse_transform(predictions)

        return {"predictions": predictions.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/arima")
def predict_arima(input_data: InputData):
    """
    Predict using the ARIMA model.
    """
    if arima_model is None:
        raise HTTPException(status_code=500, detail="ARIMA model not loaded.")
    try:
        input_array = np.array(input_data.data)
        predictions = arima_model.forecast(steps=len(input_array))

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Model inference API is running."}
