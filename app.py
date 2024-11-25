import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# API Base URL
API_BASE_URL = "http://127.0.0.1:8000"

# UI Setup
st.set_page_config(
    page_title="☕ Café ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styles
st.markdown("""
    <style>
        body {
            background-color: #f4f1ea; /* Cream color */
            color: #4e342e; /* Espresso brown */
        }
        .stButton>button {
            background-color: #4e342e;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stButton>button:hover {
            background-color: #3e2723;
            color: white;
        }
        .sidebar .sidebar-content {
            text-align: center;
        }
        .reportview-container .main .block-container {
            max-width: 90%;
            padding-top: 2rem;
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 18px;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>☕ Café ML Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Analyze and Predict Revenue & Product Performance</div>", unsafe_allow_html=True)

# Layout Setup
# Grid Layout
col1, col2 = st.columns([1, 2])  # Sidebar occupies 1/3 and main content 2/3

# Sidebar for User Input
with col1:
    st.sidebar.title("⚙️ Settings")

    # Text Input for Data Entry
    st.sidebar.markdown("### Enter Sales Data")
    text_input = st.sidebar.text_area(
        "Paste your sales data or product details:",
        placeholder="Enter data in plain text..."
    )
    
    # Dropdown for Model Selection
    model_type = st.sidebar.selectbox("Select Prediction Model", ["LSTM", "Transformer", "ARIMA"])
    
    # Option to Upload CSV
    uploaded_file = st.sidebar.file_uploader("Or Upload Test Data (CSV)", type=["csv"])

    # Process Text Input Using Gemini API (or equivalent)
    if st.sidebar.button("Prepare Input Data"):
        if text_input.strip():
            # Simulate Gemini API for preprocessing
            try:
                response = requests.post(f"{API_BASE_URL}/process", json={"text": text_input})
                response.raise_for_status()
                prepared_data = response.json()["prepared_data"]
                st.sidebar.success("Data processed successfully!")
                st.sidebar.write(prepared_data)  # Display prepared data
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Error processing data: {e}")
        else:
            st.sidebar.warning("Please enter text data or upload a file.")

# Main Content Area
with col2:
    # Prediction Results Section
    st.markdown("### 🎯 Predictions")
    if st.button("Run Prediction"):
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            input_data = {"data": data.values.flatten().tolist()}
        elif "prepared_data" in locals():
            input_data = {"data": prepared_data}
        else:
            st.error("No data available for prediction. Please upload or process data first.")
            st.stop()

        # Call Prediction API
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
                st.success("Prediction successful!")

                # Show Predictions
                st.write("### Prediction Results")
                st.write(pd.DataFrame(predictions, columns=["Predicted Values"]))

                # Reserve Space for Visualization
                st.write("### Visualizations")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=input_data["data"], name="Actual Data", mode="lines"))
                fig.add_trace(go.Scatter(y=predictions, name="Predictions", mode="lines"))
                fig.update_layout(
                    title="Actual vs Predicted Data",
                    xaxis_title="Time Steps",
                    yaxis_title="Values",
                    template="plotly_white"
                )
                st.plotly_chart(fig)
            except requests.exceptions.RequestException as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Invalid model type selected.")

    # Placeholder for Future Graphs
    st.markdown("### 📊 Future Graph Visualizations")
    st.info("Graph visualization space reserved for future updates.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### Made with ❤️ for CaféCast by [Your Team Name]")
