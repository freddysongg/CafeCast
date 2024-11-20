# CaféCast ☕📊

**CaféCast** is an AI-powered sales forecasting and product analysis application. This project focuses on leveraging advanced machine learning models and data visualization techniques to explore various datasets, fine-tune hyperparameters, and analyze complex temporal patterns. While not optimized for a production-grade frontend, the application demonstrates expertise in applying cutting-edge algorithms and interpretability techniques to time series forecasting and recommendation tasks.

---

## Features 🚀

- **Sales Forecasting:**
  - Advanced LSTM-based forecasting with Bayesian optimization and iterative tuning approaches.
  - State-of-the-art Time Series Transformers for handling complex temporal dependencies.
  - Traditional ARIMA modeling for interpretable short-term forecasts.
- **Data Visualization:** Insights into sales trends, seasonality, and product demand patterns.
- **Explainable AI:** Integration of SHAP values and attention mechanisms for model interpretability.
- **Customization-Ready:** Easily adaptable for analyzing and forecasting data across various café datasets.

---

## Models and Methodologies 📘

### 1. **LSTM with Bayesian Optimization**
- Utilizes Bayesian optimization to automatically tune critical hyperparameters such as:
  - Learning rate
  - Number of layers
  - Neurons per layer
  - Dropout rates
- This approach balances exploration and exploitation, ensuring optimal configurations with reduced computational overhead.

### 2. **Iterative LSTM Tuning**
- Applies a manual, systematic method to refine model performance through:
  - Adjustments to sequence length and batch size
  - Monitoring validation error trends
  - Incremental fine-tuning of parameters based on observed performance
- Ensures the LSTM models are tailored to the unique characteristics of each dataset.

### 3. **Time Series Transformers**
- Implements Transformer-based models for time series forecasting, leveraging self-attention mechanisms to:
  - Capture long-term temporal dependencies effectively
  - Model complex seasonality and trends in data
  - Provide highly interpretable attention weights for feature importance

### 4. **ARIMA Model**
- Integrates an ARIMA (AutoRegressive Integrated Moving Average) model for classical time series analysis.
- Features:
  - Strong interpretability for short-term linear trends and seasonality
  - Complements deep learning models for robust hybrid forecasting strategies

By combining these methodologies, CaféCast excels in flexibility and precision, adapting seamlessly to varying datasets.

---

## Tech Stack 🛠️

- **Programming Language:** Python
- **Core Libraries:**
  - NumPy, Pandas for data manipulation
  - Matplotlib for visualization
  - TensorFlow & PyTorch for deep learning
  - scikit-learn and statsmodels for preprocessing and ARIMA modeling
- **Optimization Techniques:**
  - Bayesian Optimization for automated hyperparameter tuning
  - Iterative tuning for manual refinement of LSTM models
- **Model Interpretability:** SHAP, Attention Mechanisms
- **Environment:** Designed for local execution on M1 Pro MacBook Pro or similar hardware

---

## Installation 💻

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cafecast.git
   cd cafecast
2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt

---

## Usage 💡
1. **Load Dataset:** Start by uploading your time series dataset in CSV format.
2. **Choose a Model:** Options include:
    - LSTM with Bayesian Optimization for automated fine-tuning and precision forecasts.
    - Iterative LSTM for a hands-on, customized modeling experience.
    - Time Series Transformers for advanced temporal analysis with attention mechanisms.
    - ARIMA for interpretable, classical time series modeling.
3. **Run Forecasts:** Generate detailed predictions for various time frames and visualize results.
4. **Analyze Results:** Use visualizations and SHAP-based interpretability tools to gain insights into trends and model behavior.

---

## Project Structure 📂
```plaintext
cafecast/
├── data/               # Sample datasets and preprocessing scripts
├── logs/               # Logging files for training and debugging
├── models/             # Model definitions and training scripts
├── notebooks/          # Jupyter notebooks for exploration and prototyping
├── params/             # Hyperparameter files and configurations
├── venv/               # Virtual environment for dependency management
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## License 📜
This project is licensed under the MIT License. See the LICENSE file for details.

