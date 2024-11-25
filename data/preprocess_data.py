import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import os

def prepare_data(file_path, date_col='transaction_date', time_col='transaction_time'):
    """
    Prepares the data by processing date and time columns.

    Args:
        file_path (str): Path to the input Excel file.
        date_col (str): Name of the date column.
        time_col (str): Name of the time column.

    Returns:
        pd.DataFrame: Processed DataFrame with date and time columns handled.
    """
    data = pd.read_excel(file_path, engine='openpyxl')
    
    data[date_col] = pd.to_datetime(data[date_col])
    
    data.set_index(date_col, inplace=True)
    
    if time_col in data.columns:
        data['transaction_hour'] = data[time_col].apply(lambda x: x.hour)
    
    return data

def generate_test_data(file_path, output_file='test_payload.json', seq_length=10):
    """
    Generates test data payloads for model inference.

    Args:
        file_path (str): Path to the input Excel file.
        output_file (str): Path to save the JSON test payload.
        seq_length (int): Length of sequences for test data.

    Returns:
        None: Saves the test payload as a JSON file.
    """
    data = prepare_data(file_path)
    
    daily_data = data.resample('D')['transaction_qty'].sum()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))
    
    sequences = [
        scaled_data[i:i + seq_length].flatten().tolist()
        for i in range(len(scaled_data) - seq_length)
    ]
    
    if sequences:
        test_data = {"data": sequences[0]}  
        
        with open(output_file, 'w') as f:
            json.dump(test_data, f)
        
        print(f"Test data saved to {output_file}")
    else:
        print("Not enough data to generate sequences. Please ensure the dataset is sufficient.")

def correlate_product_info(input_file, output_file):
    """
    Correlates product_id with product_category, product_type, and product_detail,
    and creates a mapping table.

    Args:
        input_file (str): Path to the input CSV file with coffee shop data.
        output_file (str): Path to save the updated CSV file with correlated product info.

    Returns:
        None: Saves the updated file to `output_file`.
    """
    data = pd.read_csv(input_file)

    required_columns = ['product_id', 'product_category', 'product_type', 'product_detail']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")

    product_mapping = data[['product_id', 'product_category', 'product_type', 'product_detail']].drop_duplicates()

    mapping_file = os.path.join(os.path.dirname(output_file), 'product_mapping.csv')
    product_mapping.to_csv(mapping_file, index=False)
    print(f"Product mapping saved to {mapping_file}")

    data = pd.merge(data, product_mapping, on='product_id', how='left')

    data.to_csv(output_file, index=False)
    print(f"Updated dataset saved to {output_file}")

def process_data(file_path, output_path):
    """
    Processes the data by adding new features and saving the result.

    Args:
        file_path (str): Path to the input file.
        output_path (str): Path to save the processed file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    data = pd.read_csv(file_path)

    data['transaction_date'] = pd.to_datetime(data['transaction_date'])

    data['date'] = data['transaction_date'].dt.day

    data['month'] = data['transaction_date'].dt.month_name()

    data['transaction_time'] = pd.to_datetime(data['transaction_time'], format='%H:%M:%S')
    data['time_distribution'] = pd.cut(
        data['transaction_time'].dt.hour,
        bins=[0, 6, 9, 12, 15, 18, 21, 24],
        labels=["0–6", "6–9", "9–12", "12–15", "15–18", "18–21", "21–24"],
        right=False,
        include_lowest=True
    )

    data['revenue'] = data['transaction_qty'] * data['unit_price']

    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return data

if __name__ == "__main__":
    input_file = 'data/cafecast_data.xlsx' 
    output_file = 'data/processed_coffee_shop_data.csv'
    product_id_output_file = 'data/product_correlation_data.csv'

    # processed_data = process_data(input_file, output_file)
    # print(processed_data.head())

    # test_output = 'test_payload.json'
    # generate_test_data(input_file, test_output)

    correlate_product_info(output_file, product_id_output_file)
