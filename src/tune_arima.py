import warnings
warnings.filterwarnings("ignore", "urllib3 v2 only supports OpenSSL")

import pmdarima as pm
import pandas as pd
from data.modify_dataset import prepare_data

data = prepare_data('data/cafecast_data.xlsx')
daily_data = data.resample('D')['transaction_qty'].sum()

# Train-test split
train_size = int(len(daily_data) * 0.8)
train, test = daily_data[:train_size], daily_data[train_size:]

# Use auto_arima to find the optimal (p, d, q)
model = pm.auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    trace=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5, max_q=5,
    max_d=2
)

print(f'Optimal ARIMA Order: {model.order}')