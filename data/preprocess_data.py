
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

def prepare_data(file_path, date_col='transaction_date', time_col='transaction_time'):
    data = pd.read_excel(file_path, engine='openpyxl')
    
    # Convert date column to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Set the date column as the index
    data.set_index(date_col, inplace=True)
    
    # Extract transaction hour if the time column exists
    if time_col in data.columns:
        data['transaction_hour'] = data[time_col].apply(lambda x: x.hour)
    
    return data

def generate_test_data(file_path, output_file='test_payload.json', seq_length=10):
    # Prepare data
    data = prepare_data(file_path)
    
    # Resample to daily transaction quantities
    daily_data = data.resample('D')['transaction_qty'].sum()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))
    
    # Generate sequences
    sequences = [
        scaled_data[i:i + seq_length].flatten().tolist()
        for i in range(len(scaled_data) - seq_length)
    ]
    
    # Create a sample payload with the first sequence
    if sequences:
        test_data = {"data": sequences[0]}  # Taking the first sequence for testing
        
        # Save the test payload to a JSON file
        with open(output_file, 'w') as f:
            json.dump(test_data, f)
        
        print(f"Test data saved to {output_file}")
    else:
        print("Not enough data to generate sequences. Please ensure the dataset is sufficient.")

def process_data(file_path, output_path):
    data = pd.read_csv(file_path)

    # Convert `transaction_date` to datetime
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])

    # Add new columns
    # 1. Date (Day)
    data['date'] = data['transaction_date'].dt.day

    # 2. Month (Full Month Name)
    data['month'] = data['transaction_date'].dt.month_name()

    # 3. Time distribution (Group times into categories)
    data['transaction_time'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S')
    data['time_distribution'] = pd.cut(
        data['transaction_time'].dt.hour,
        bins=[0, 6, 9, 12, 15, 18, 21, 24],
        labels=["0–6", "6–9", "9–12", "12–15", "15–18", "18–21", "21–24"],
        right=False,
        include_lowest=True
    )

    # 4. Revenue (transaction_qty * unit_price)
    data['revenue'] = data['transaction_qty'] * data['unit_price']

    # Save the processed data to a new file
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return data

if __name__ == "__main__":
    input_file = 'data/cafecast_data.xlsx' 
    output_file = 'data/processed_coffee_shop_data.csv'  # Replace with your desired output file

    processed_data = process_data(input_file, output_file)
    print(processed_data.head())

    # Example usage for generating test data
    excel_file = 'data/cafecast_data.xlsx'  
    test_output = 'test_payload.json'
    generate_test_data(excel_file, test_output)
