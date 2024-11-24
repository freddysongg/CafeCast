
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def evaluate_predictions(actuals, predictions):
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def plot_predictions(actuals, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", marker='o')
    plt.plot(predictions, label="Predicted", marker='x')
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

def naive_forecast(actuals):
    return actuals[:-1]  # Predict the last observed value for all steps

def moving_average_forecast(actuals, window=3):
    return [np.mean(actuals[i-window:i]) for i in range(window, len(actuals))]

if __name__ == "__main__":
    # Example usage:
    # Replace these with your actual test and prediction data
    actual_values = [100, 105, 110, 120]  # Example actual values
    predicted_values = [98, 107, 115, 118]  # Example predicted values

    # Evaluate metrics
    metrics = evaluate_predictions(actual_values, predicted_values)
    print("Evaluation Metrics:", metrics)

    # Plot actual vs predicted values
    plot_predictions(actual_values, predicted_values)

    # Baselines
    naive = naive_forecast(actual_values)
    moving_avg = moving_average_forecast(actual_values, window=2)
    print("Naive Forecast:", naive)
    print("Moving Average Forecast:", moving_avg)
