import pandas as pd
import numpy as np
import math
import warnings

from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


class ARIMAModel:
    """Class method to compute ARIMA model for the Stock prices."""

    def __init__(self, p, d, q, horizon):
        self.p = p
        self.d = d
        self.q = q
        self.horizon = horizon

    def end_training_data(self, current_k_fold, end_value_fold):
        self.end_training_data = end_value_fold * current_k_fold
        self.end_test_data = self.end_training_data + self.horizon
    
    def prepare_train_test_data(self, df, feature):
        self.training_data = df[0:self.end_training_data][feature].values
        self.test_data = df[self.end_training_data:self.end_test_data][feature].values
    
    def prepare_data_train(self, df, feature):
        self.training_data = df[feature]
    
    def fit_predict(self, horizon=None):
        self.model = ARIMA(self.training_data, order=(self.p, self.d, self.q)).fit(disp=0)
        output = self.model.forecast(steps=horizon if horizon else self.horizon) ## Predict Horizon days after model applied.
        self.predictions = output[0]

    def compute_rmse_metric(self):
        MSE_error = mean_squared_error(self.test_data, self.predictions)
        RMSE_error = math.sqrt(MSE_error)

        return RMSE_error

    

class ProphetModel:
    """Class method to compute Prophet model for the Stock prices."""

    def __init__(self,horizon):
        self.horizon = horizon

    def prepare_dataset(self, df, feature):
        # Prophet needs the data in a specific format.
        df = df[feature].reset_index()
        self.df = df.rename(columns = {"Date":"ds", feature:"y"}) #renaming the columns of the dataset
        self.feature = 'y'
        
    def end_training_data(self, current_k_fold, end_value_fold):
        self.end_training_data = end_value_fold * current_k_fold
        self.end_test_data = self.end_training_data + self.horizon
    
    def prepare_train_test_data(self):
        self.training_data = self.df[0:self.end_training_data]
        self.test_data = self.df[self.end_training_data:self.end_test_data]
    
    def fit_predict(self, daily_seasonality = True, yearly_seasonality = True):
        model = Prophet(daily_seasonality = daily_seasonality, yearly_seasonality = yearly_seasonality).fit(self.training_data)
        future = model.make_future_dataframe(periods=self.horizon)
        #we need to specify the number of days in future
        self.predictions = model.predict(future)
        
    def compute_rmse_metric(self):
        MSE_error = mean_squared_error(self.test_data['y'], self.predictions[self.end_training_data:self.end_test_data]['yhat'])
        RMSE_error = math.sqrt(MSE_error)

        return RMSE_error   

    

class LSTMModel:
    """Class method to compute LSTM model for the Stock prices."""
    
    def __init__(self, horizon=5, time_steps=15, units=30, dropout=0.2, epochs=100, optimizer='adam', loss='mean_squared_error', batch_size=32, activation='sigmoid'):

        self.time_steps = time_steps
        self.units = units
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.activation = activation
        self.model = Sequential()  
        
    def end_training_data(self, current_k_fold, end_value_fold):
        self.end_training_data = end_value_fold * current_k_fold
        self.end_test_data = self.end_training_data + self.horizon
    
    def prepare_train_test_data(self, df):
        training_set = df.iloc[:self.end_training_data, 0:1]
        test_set = df.iloc[self.end_training_data:self.end_test_data, 0:1]

        # Feature Scaling
        self.sc = MinMaxScaler(feature_range = (0, 1)).fit(training_set.values)
        self.training_set_scaled = self.sc.transform(training_set.values)

        dataset_total = pd.concat((training_set, test_set), axis = 0)
        test_df = dataset_total[len(dataset_total) - len(test_set) - self.time_steps:].values
        test_df = test_df.reshape(-1,1)
        self.test_set_scaled = self.sc.transform(test_df)
    
    def build_train_test_d(self):
    # Creating a data structure with X time-steps and 1 output
        X_train, y_train, X_test, y_test = [], [], [], [] # we create sliding window sets to predict from previous time-steps days, the next day.

        for i in range(0, len(self.training_set_scaled) - self.time_steps - self.horizon):
            X_train.append(self.training_set_scaled[i:i + self.time_steps, 0])
            y_train.append(self.training_set_scaled[i + self.time_steps:i + self.time_steps + self.horizon, 0])

        for i in range(0, 1): # just test the last value as in the other models
            X_test.append(self.test_set_scaled[i:i + self.time_steps, 0])
            y_test.append(self.test_set_scaled[i + self.time_steps:i + self.time_steps + self.horizon,0])

        self.X_train, self.y_train, self.X_test, self.y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (1, self.time_steps, 1))

        # Shape (#values, #time-steps, #horizon).
#         print(f"* X_training shape={X_train.shape} and y_train={y_train.shape} \n* X_test shape={X_test.shape} and y_test={y_test.shape}\n")  
        
    def lstm_1_layer(self):
        # vanilla LSTM with just one LSTM layer
        self.model.add(LSTM(units = self.units, return_sequences = False, input_shape = (self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropout))        

    def lstm_2_layers(self):
        # vanilla LSTM with just one LSTM layer - 1st Layer
        self.model.add(LSTM(units = self.units, return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropout))        
        # 2nd layer
        self.model.add(LSTM(units = self.units))
        self.model.add(Dropout(self.dropout))    
        
    def lstm_3_layers(self):
        # vanilla LSTM with just one LSTM layer - 1st Layer
        self.model.add(LSTM(units = self.units, return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropout))        
        # 2nd layer
        self.model.add(LSTM(units = self.units, return_sequences = True))
        self.model.add(Dropout(self.dropout))
        # 3rd layer
        self.model.add(LSTM(units = self.units))
        self.model.add(Dropout(self.dropout))    
    
    def lstm_4_layers(self):
        # vanilla LSTM with just one LSTM layer - 1st Layer
        self.model.add(LSTM(units = self.units, return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
        self.model.add(Dropout(self.dropout))        
        # 2nd layer
        self.model.add(LSTM(units = self.units, return_sequences = True))
        self.model.add(Dropout(self.dropout))
        # 3rd layer
        self.model.add(LSTM(units = self.units, return_sequences = True))
        self.model.add(Dropout(self.dropout))
        # 4th layer
        self.model.add(LSTM(units = self.units))
        self.model.add(Dropout(self.dropout))    
    
    def lstm_bidirectional_layer(self):
        self.model.add(Bidirectional(LSTM(self.units, return_sequences=True), input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Bidirectional(LSTM(self.units)))

    def dense_layer(self, units):
        self.model.add(Dense(units=units if units else self.units, activation=self.activation))
        self.model.add(Dense(units = self.horizon))
        
    def compile_fit_rnn(self):
        # Compiling the RNN
        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        # Fitting the RNN to the Training set
        self.model.fit(self.X_train, self.y_train, epochs = self.epochs, batch_size = self.batch_size, verbose=0)
        
    def predict_invert_values(self):
        self.predictions_inverted = self.sc.inverse_transform(self.model.predict(self.X_test))
        self.y_test_inverted = self.sc.inverse_transform(self.y_test)
        
    def compute_rmse_metric(self):
        MSE_error = mean_squared_error(self.y_test_inverted, self.predictions_inverted)
        RMSE_error = math.sqrt(MSE_error)

        return RMSE_error   
