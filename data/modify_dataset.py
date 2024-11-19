import pandas as pd

def prepare_data(file_path, date_col='transaction_date', time_col='transaction_time'):
    data = pd.read_excel(file_path)
    
    data[date_col] = pd.to_datetime(data[date_col])
    
    data.set_index(date_col, inplace=True)
    
    if time_col in data.columns:
        data['transaction_hour'] = data[time_col].apply(lambda x: x.hour)
    
    return data

if __name__ == "__main__":
    data = prepare_data('../data/cafecast_data.xlsx')
    print(data.info())  
