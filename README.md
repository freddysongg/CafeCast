# CaféCast ☕📊

**CaféCast** is a data-focused platform designed to help users explore and analyze datasets using AI-driven insights. This project currently focuses on manual data cleaning, shaping, and model training workflows, with plans to evolve into a fully interactive and user-friendly platform.

---

## Features 🚀

- **Cross-Platform Model Execution:**
  - Supports **TensorFlow (CPU)** for macOS users to ensure compatibility and efficient local execution.
  - Utilizes **PyTorch (CUDA)** for Windows users with GPU support for accelerated training and predictions.
  - Automatic detection of the operating system to ensure the correct model implementation is selected.
- **Data Cleaning and Shaping:**
  - Manual workflows for preparing datasets for analysis.
- **Model Training:**
  - Advanced LSTM-based models with Bayesian optimization and iterative tuning for precise forecasts.
  - Time Series Transformers for capturing complex temporal dependencies.
  - Classical ARIMA modeling for interpretable, short-term linear trend forecasts.
- **Enhanced Model Flexibility:**
  - Configurable hyperparameters to suit the unique characteristics of each dataset.
  - Dynamic support for both platform-optimized and manually fine-tuned workflows.
- **Data Visualization:** Visual insights into sales trends, seasonality, and demand patterns.

---

## Purpose and Motivation 🎯

The primary objective of **CaféCast** is to test my knowledge, challenge myself, and practice advanced machine learning techniques. Through this project, I aim to:

- **Deepen Understanding:** Dive into various machine learning models, including LSTM, Transformers, and ARIMA.
- **Refine Skills:** Gain hands-on experience in hyperparameter fine-tuning using Bayesian optimization and manual iterative tuning.
- **Explore Interpretability:** Learn and apply techniques like SHAP values and attention mechanisms to make models more transparent.
- **Emphasize Learning:** Approach this as a learning process, with the goal of improving my practical skills in applying machine learning models to real-world forecasting tasks.

This project is a testament to continuous learning and experimentation in the field of AI and machine learning.

---

## Models and Methodologies 📘

### 1. **LSTM with Bayesian Optimization**
- Automatically tunes hyperparameters, including:
  - Learning rate
  - Number of layers and neurons per layer
  - Dropout rates
- Strikes a balance between exploration and exploitation to achieve efficient and optimal configurations.

### 2. **Iterative LSTM Tuning**
- A hands-on approach for refining models through:
  - Adjustments to sequence length and batch size
  - Monitoring validation error trends
  - Incremental parameter tuning based on observed performance

### 3. **Time Series Transformers**
- Uses self-attention mechanisms to:
  - Capture long-term dependencies and seasonality
  - Model complex temporal patterns in sales data
- Supports multi-target predictions for key metrics like sales and revenue.

### 4. **ARIMA Model**
- A classical time series model for capturing linear trends and seasonality.
- Complements deep learning models for hybrid forecasting strategies.

---

## Recent Enhancements 🌟

1. **Platform-Specific Execution:**
   - macOS: TensorFlow-based models optimized for CPU execution.
   - Windows: PyTorch-based models leveraging CUDA for GPU acceleration.
   - Automatic logging to indicate which implementation is running.
   - Note: this is all for training of the model, mainly just convenience for me :D

2. **Broader Prediction Capabilities:**
   - Added support for handling multiple predictions across various datasets.
   - Enhanced flexibility to adapt to different temporal forecasting requirements.

3. **Improved Model Interpretability:**
   - SHAP values for LSTM and ARIMA models to explain predictions.
   - Attention weights in Transformer models to provide insights into feature importance.

---

## Future Expansions 🚀

CafeCast is set to evolve into a more interactive and user-friendly platform with the following key features:

### 1. **User-Focused Frontend Deployment**

- A dedicated web-based interface for seamless interaction.
- Users can upload datasets directly through the frontend, removing the need for manual backend processes.
- A clean and intuitive design that prioritizes simplicity and accessibility.
- Technologies considered include **React**, **Next.js**, or other modern web frameworks for dynamic and responsive user experiences.

### 2. **AI-Driven Q&A Functionality**

- Users will be able to ask questions about their uploaded data through the frontend.
- The platform will leverage pre-trained models saved on the backend to provide accurate and meaningful answers.
- Natural language processing (NLP) capabilities will interpret user queries and deliver real-time responses.
- Support for follow-up queries to explore datasets more deeply.

### 3. **Automated Data Cleaning and Preprocessing**

- Automate the current manual data cleaning workflow.
- The platform will:
  - Validate datasets upon upload.
  - Detect and resolve missing values, outliers, and inconsistencies automatically.
  - Present a preview of cleaned data for user confirmation.

### 4. **Model Training and Customization**

- Allow users to train or fine-tune models directly through the platform using their uploaded datasets.
- Include visualization tools for monitoring training progress, accuracy, and performance metrics.
- Ensure that users can easily customize their models to suit specific use cases.

---

## Tech Stack 🛠️

- **Programming Language:** Python
- **Core Libraries:**
  - NumPy, Pandas for data manipulation
  - Matplotlib for visualization
  - TensorFlow (CPU) for macOS
  - PyTorch (CUDA) for Windows
  - scikit-learn and statsmodels for preprocessing and ARIMA modeling
- **Optimization Techniques:**
  - Bayesian Optimization for automated hyperparameter tuning
  - Iterative tuning for manual refinement of models
- **Model Interpretability:** SHAP, Attention Mechanisms
- **Environment:** Tested on macOS and Windows with hardware-optimized configurations

---

## Installation 💻

1. Clone the repository:
   ```bash
   git clone https://github.com/freddysongg/cafecast.git
   cd cafecast
2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt # On Windows: add in pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 for Torch-CUDA compatability

---

## Usage 💡
1. **Run the Application:** 
    ```bash
    python src/main.py
2. **Menu Options:**
    1: Run LSTM Model
    2: Run Time Series Transformer Model
    3: Run Bayesian LSTM Optimization
    4: Clear LSTM Model Parameters
    5: Clear Transformer Model Parameters
    6: Run ARIMA Model
3. **Run Forecasts:** Generate detailed predictions for various time frames and visualize results.
4. **Analyze Results:** Use visualizations and SHAP-based interpretability tools to gain insights into trends and model behavior.

---

## Project Structure 📂
```plaintext
cafecast/
├── data/               # Sample datasets and preprocessing scripts
├── logs/               # Logging files for training and debugging
├── models/             # Model definitions and training scripts
├── params/             # Hyperparameter files and configurations
├── src/                # Main application code and model implementations
├── venv/               # Virtual environment for dependency management
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## License 📜
This project is licensed under the MIT License. See the LICENSE file for details.

