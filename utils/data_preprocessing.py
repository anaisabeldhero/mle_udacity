import yfinance as yf
import pandas as pd
import pickle

from utils.data.data_config import LIST_DAX_COMPANIES


def download_stocks_from_list(start_date='2016-01-01', end_date="2021-12-31"):
    """
    Download the data for all the stocks in the list and save the data as a CSV.
    :params start_date: datetime in format 'YYYY-MM-DD', start limit to read data from.
    :params end_date: datetime in format 'YYYY-MM-DD', end limit to read data from.
    """
    df = yf.download(" ".join(LIST_DAX_COMPANIES),
                     start=start_date,
                     end=end_date)
    
    save_df_and_columns_name(df, 'utils/data/data_30_2016.csv', 'utils/data/column_file.py')
    
    return df


def save_df_and_columns_name(df, csv_name, columns_file):
    """
    Save the Dataframe into csv and columns in a list in python script..
    :params df: dataframe with the downloaded data from yfinance.
    :params csv_name: name of the csv to save.
    :params columns_file: script name with the columns config.
    """
    df.to_csv(csv_name, header=False) # save data in csv

    columns_name = []
    for multi_col in df.columns:
        columns_name.append(multi_col[0] + '_' + multi_col[1])      
    
    with open(columns_file, 'wb') as f:
        pickle.dump(columns_name, f)


def read_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        file_read = pickle.load( f)
    
    return file_read


def read_csv_all_stocks(columns_name, csv_file_name):
    """
    Read from a csv all the stocks from DAX. The csv name is 'dax_30_2016.csv' and includes data from all companies in DAX's index.
    :params columns_name: name of the list with the columns name.
    :params csv_file_name: name of the csv to read.
    """
    columns = []
    for column in columns_name:
        col = column.split('_')
        columns.append((col[0], col[1]))

    df = pd.read_csv(csv_file_name,  names=columns)
    df.index.set_names('Date', inplace=True)
    
    return df


def prepare_df(start_date='2016-01-01', end_date="2021-12-31"):
    """
    Ensure the data is in the format required for the analysis
    :params start_date: datetime in format 'YYYY-MM-DD', start limit to read data from.
    :params end_date: datetime in format 'YYYY-MM-DD', end limit to read data from.
    """
    df = read_csv_all_stocks(read_pickle_file('utils/data/column_file.py'), 'utils/data/data_30_2016.csv')
    
    df = df.swaplevel(i=- 2, j=- 1, axis=1) # swap levels to be able to call the symbol directly.
    # ensure we only have data from the provided dates.
    df = df[(df.index >= start_date) & (df.index <= end_date)] # remove data from 2015

    # ensure we have daily sample, including the weekend that are missing or holidays date
    df.asfreq('D') # just with this we ensure we get a 'Daily' frequency, as previously it was set up automatically to 'None'
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')    
    
    return df


def impute_data_points(df, interpolation_method='linear'):
    """Impute values to the missing dates. We can use the methods linear or quadratic.
    - Linear interpolates values from a linear equation 
    - Quadratic interpolates values from a quadratic equation."""
    
    for column in df.columns:
        df[column] = df[column].interpolate(method=interpolation_method)
        
    return df


