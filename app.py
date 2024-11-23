import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import tensorflow as tf
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Helper functions
def load_scaler(path):
    with open(path, 'r') as f:
        scaler_data = json.load(f)
    scaler = MinMaxScaler()
    scaler.min_ = np.array(scaler_data['min_'])
    scaler.scale_ = np.array(scaler_data['scale_'])
    return scaler

def load_lstm_model():
    return tf.keras.load_model('models/best_lstm_model.keras')

def load_transformer_model(params_path, model_path):
    with open(params_path, 'r') as f:
        params = json.load(f)
    model = TimeSeriesTransformer(
        input_size=params['d_model'],
        num_layers=params['num_layers'],
        num_heads=params['num_heads'],
        d_model=params['d_model'],
        dim_feedforward=params['dim_feedforward']
    )
    model.load_state_dict(torch.load(model_path))
    return model, params

# TimeSeriesTransformer class definition (same as in your training code)
class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, input_size, num_layers, num_heads, d_model, dim_feedforward):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Output of the last time step
        return x

# UI Setup
st.set_page_config(
    page_title="Café ML Demo",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Style Settings
st.markdown("""
    <style>
        body {
            background-color: #f4f1ea; /* Cream color */
            color: #4e342e; /* Espresso brown */
        }
        .stButton>button {
            background-color: #4e342e;
            color: white;
        }
        .stButton>button:hover {
            background-color: #6d4c41;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Café ML Demo")
st.sidebar.title("⚙️ Settings")

# Sidebar options
model_type = st.sidebar.selectbox("Select Model Type", ["LSTM", "Transformer"])
seq_length = st.sidebar.number_input("Sequence Length", min_value=5, max_value=50, value=10, step=1)
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
dark_mode = st.sidebar.checkbox("Enable Dark Mode")

# Dark Mode Styling
if dark_mode:
    st.markdown("""
        <style>
            body {
                background-color: #2c2c2c;
                color: #f4f1ea;
            }
            .stButton>button {
                background-color: #6d4c41;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# Load Data
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.write(f"Data preview:")
    st.sidebar.write(data.head())
else:
    st.sidebar.warning("Upload a CSV file to proceed.")

# Load Models
scaler_path = 'models/scaler.json'
scaler = load_scaler(scaler_path)
lstm_model = None
transformer_model = None
if model_type == "LSTM":
    lstm_model = load_lstm_model()
else:
    transformer_model, transformer_params = load_transformer_model(
        'params/best_ts_transformer_params.json',
        'models/best_ts_transformer_model.pt'
    )

# Inference
if st.button("Run Inference"):
    if not uploaded_file:
        st.error("Please upload a test data file first.")
    else:
        # Scale and process data
        scaled_data = scaler.transform(data.values)
        sequences = [
            scaled_data[i : i + seq_length]
            for i in range(len(scaled_data) - seq_length)
        ]
        sequences = np.array(sequences)

        # Predict
        if model_type == "LSTM":
            predictions = lstm_model.predict(sequences)
        else:
            sequences_torch = torch.FloatTensor(sequences).unsqueeze(-1)  # Add feature dim
            predictions = transformer_model(sequences_torch).detach().numpy()

        # Rescale predictions
        predictions_rescaled = scaler.inverse_transform(predictions)

        # Visualization
        st.success("Inference complete! Here are the results:")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data.values.flatten(), name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(y=predictions_rescaled.flatten(), name="Predicted", mode="lines"))
        fig.update_layout(
            title="Actual vs Predicted",
            xaxis_title="Time Steps",
            yaxis_title="Values",
            template="plotly_dark" if dark_mode else "plotly_white",
        )
        st.plotly_chart(fig)

# Footer
st.markdown("#### Made with ❤️ for CaféCast")
