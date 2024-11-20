import sys

def run_arima():
    import arima
    arima.main()

def run_lstm():
    import lstm
    lstm.main()

def run_transformer():
    import ts_transformer
    ts_transformer.main()

if __name__ == "__main__":
    print("Choose a model to run:")
    print("1. ARIMA")
    print("2. LSTM RNN")
    print("3. Time Series Transformer")
    print("4. Custom Pipeline (all models)")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        run_arima()
    elif choice == '2':
        run_lstm()
    elif choice == '3':
        run_transformer()
    elif choice == '4':
        print("Running a custom pipeline combining ARIMA, LSTM, and Transformer...")
        run_arima()
        run_lstm()
        run_transformer()
    else:
        print("Invalid choice. Please select 1, 2, 3, or 4.")
