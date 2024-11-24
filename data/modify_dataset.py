import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path, date_col='transaction_date', time_col='transaction_time'):
    data = pd.read_excel(file_path)
    
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

if __name__ == "__main__":
    file_path = 'data/cafecast_data.xlsx'  
    data = prepare_data(file_path)
    print(data.info()) 
    
    # Generate and save test data
    generate_test_data(file_path, output_file='test_payload.json', seq_length=10)
