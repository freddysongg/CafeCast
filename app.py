import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# API Base URL 
API_BASE_URL = "http://127.0.0.1:8000"

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
model_type = st.sidebar.selectbox("Select Model Type", ["LSTM", "Transformer", "ARIMA"])
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
data = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("Data preview:")
    st.sidebar.write(data.head())
else:
    st.sidebar.warning("Upload a CSV file to proceed.")

# Run Inference
if st.button("Run Inference"):
    if data is None:
        st.error("Please upload a test data file first.")
    else:
        # Convert data to JSON-friendly format
        input_data = {"data": data.values.flatten().tolist()}

        # Call API based on model type
        endpoint = {
            "LSTM": "/predict/lstm",
            "Transformer": "/predict/transformer",
            "ARIMA": "/predict/arima"
        }.get(model_type)

        if endpoint:
            try:
                response = requests.post(API_BASE_URL + endpoint, json=input_data)
                response.raise_for_status()
                predictions = response.json()["predictions"]

                # Visualization
                st.success("Inference complete! Here are the results:")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=data.values.flatten(), name="Actual", mode="lines"))
                fig.add_trace(go.Scatter(y=predictions, name="Predicted", mode="lines"))
                fig.update_layout(
                    title="Actual vs Predicted",
                    xaxis_title="Time Steps",
                    yaxis_title="Values",
                    template="plotly_dark" if dark_mode else "plotly_white",
                )
                st.plotly_chart(fig)
            except requests.exceptions.RequestException as e:
                st.error(f"API call failed: {e}")
        else:
            st.error("Invalid model type selected.")

# Footer
st.markdown("#### Made with ❤️ for CaféCast")
