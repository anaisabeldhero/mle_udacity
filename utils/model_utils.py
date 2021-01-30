""" Create functions to compose the models
    - Build training and test set
    - Fit model
    - Predict test set
    - Compute the objective metric RMSE
"""
import warnings
import os

from utils.models_class import ARIMAModel, ProphetModel, LSTMModel

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def run_arima_model(horizon, current_k, end_value_fold, df_symbol, feature, p=6, d=1, q=0):
    model = ARIMAModel(p, d, q, horizon)
    model.end_training_data(current_k + 1, end_value_fold)
    model.prepare_train_test_data(df_symbol, feature)
    model.fit_predict()
    
    return model.compute_rmse_metric()


def run_prophet_model(horizon, current_k, end_value_fold, df_symbol, feature, daily_seasonality=True, yearly_seasonality=True):
    model = ProphetModel(horizon)
    model.prepare_dataset(df_symbol, feature)
    model.end_training_data(current_k + 1, end_value_fold)
    model.prepare_train_test_data()
    model.fit_predict(daily_seasonality, yearly_seasonality)
    
    return model.compute_rmse_metric()


def run_sequential_lstm_model(horizon, current_k, end_value_fold, df_symbol, feature, model_type, time_steps=15, units=30, dropout=0.2, epochs=100):

    model = LSTMModel(horizon=horizon, time_steps=time_steps, units=units, dropout=dropout, epochs=epochs)
    model.end_training_data(current_k + 1, end_value_fold)
    model.prepare_train_test_data(df_symbol)
    model.build_train_test_d()
    
    if model_type == 'LSTM_1':
        model.lstm_1_layer()
    elif model_type == 'LSTM_2':
        model.lstm_2_layers()
    elif model_type == 'LSTM_3':
        model.lstm_3_layers()
    elif model_type == 'LSTM_4':
        model.lstm_4_layers()
    elif model_type == 'LSTM_BI':
        model.lstm_bidirectional_layer()

    model.dense_layer(units=30)  # try with 10 too.
    model.compile_fit_rnn()
    model.predict_invert_values()
    
    return model.compute_rmse_metric()


def model_decision(model_type, horizon, current_k, end_value_fold, df_symbol, feature):
    if model_type == 'ARIMA':
        return run_arima_model(horizon, current_k, end_value_fold, df_symbol, feature)
    elif model_type == 'Prophet':
        return run_prophet_model(horizon, current_k, end_value_fold, df_symbol, feature)
    else:
        return run_sequential_lstm_model(horizon, current_k, end_value_fold, df_symbol, feature, model_type)   
    

def cross_validation(data, symbol, model_type):
    df_symbol = data[symbol].copy()
    
    df_symbol.dropna(inplace=True)  # NAN values removed because DHER.DE has null values
    
    k, horizon = 8, 5
    feature = 'Adj Close'
    end_value_fold = int(len(df_symbol[feature])/k)
    rmse_error_total = []
    
    for current_k in range(k-1):
        rmse_error_total.append(model_decision(model_type, horizon, current_k, end_value_fold, df_symbol, feature))

    rsme_mean = sum(rmse_error_total)/(k-1)
    print(f"-> {model_type}: The average RMSE of symbol {symbol} of the CV with {k}-fold is: {rsme_mean}")
    
    return rsme_mean
