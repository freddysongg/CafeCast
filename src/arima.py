import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data.modify_dataset import prepare_data

# Load and prepare data
data = prepare_data('data/cafecast_data.xlsx')
daily_data = data.resample('D')['transaction_qty'].sum()

# Train-test split
train_size = int(len(daily_data) * 0.8)
train, test = daily_data[:train_size], daily_data[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(1, 1, 2))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Plot actual vs forecast
plt.figure(figsize=(14, 7))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Actual vs Forecasted Transaction Quantities (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Transaction Quantity')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate model performance
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')