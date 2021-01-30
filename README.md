# STOCK PREDICTOR PRICES FOR DAX30

## Motivation

This is the capstone project report for the Machine Learning Nanodegree from Udacity. 
We focus on predicting the stock prices for the companies in the DAX30, the blue-chip stock marketing index with the 30 major German companies trading on the Frankfurt Stock Exchange.

To provide the next best action for investing in stocks, in this project we compare the performance of 3 different models and techniques. 
The models are ARIMA, used as benchmark, Prophet and LSTM NN.
The best model is used in a dashboard to predict based on some parameters that the user should input.

## Summary

The main idea of this project is to understand what is the next best action for a given symbol or company.
To know the next best action, the stocks must be predicted. The challenge of this proposal is to build a stock 
price predictor that takes daily trading data over a certain data range as input and outputs the projected estimates
for the next 5 days. For this purpose, we will model 2 models: prophet designed by Facebook data science team and 
Neural Network LSTM model. The performance of those models are compared to a benchmark model that in our case it's 
an ARIMA model. The best model will be used in a app interface to advice the client what to do next with your current 
investment in a certain symbol or if to start investing in there.

This project has been completely developed in `python`. We have used mainly `python scripts`, `python notebook` and `dash` for creating a dashboard.

The main libraries used are:

```python
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import yfinance as yf
import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot

from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
```
Description:

1. The first block is used to create the dash dashboard. 
2. The second block is used to download the data and to do computation with index column of type datetime.
3. The third block is used to process and manipulate the dataset.
4. For visualization we use `matplotlib`.
5. The fifth block is used to create the 3 different models that we compare in this project.
6. The sixth block provides the functions to scale the data and the comparison metric.

## Installation

To run the notebooks and the app, please install the requirements first. 
Don't forget to activate your virtual environment first. Run the following command from the main directory of this repository.

```bash
pip install -r requirements.txt
```

This library runs in python 3. You could face some problems and errors if you have a different version.

## Repository Structure

This project contains different scripts and folders organized in the following way: 

1. `dashapp` folder holds the dashboard application for Prediction the stocks price. 
2. `graphs` folder contains the images attached in the report.md or report.pdf
3. `utils` folder with the functions used in the notebooks.

The analysis for this project has divided into 3 notebooks, please runs them in order to follow the report:

1. `data_exploration.ipynb` contains all the EDA of this project.
2. `hyperparamter_tuning.ipynb` holds the full process to tune the 3 proposed models in this project. 
3. `results.ipynb` contains the comparison between the models and the elaboration of the decision function for the dashboard.

**Please note:** all output cells of the notebooks are included as the `hyperpameter_tunning.ipynb` lasts around 1 day to run completely and `results.ipynb` needs around 8 hours.

The functions used in the notebooks are factorized into different scripts in order to increase the readability of the project. 

1. `models_class.py` is the script with the definition of the 3 model classes. 
2. `model_utils.py` is the script with functions built on top of the model classes for a faster usage of them.
3. `data/data_config.py` holds the list of companies that we will use in this project, the 30 companies of DAX index.
4. `data_preprocessing.py` is the script with the functions to download the data and prepare it for usage.
5. `data/column_file.py` is a output script from `data_preprocessing.py` that saves the multi-index columns of the dataset. This will be only used in case we need to work offline.
6. `data/dax_30_2016.csv` is a CSV file with the data for all the DAX companies since beginning 2016 till end of 2020.

## Usage

Launch the dashboard with the following command:

```python
python3 -m dashapp.app
```

The Dash will run on `http://127.0.0.1:8050/`, please read the output of this command to access with it to your browser.
In case of problems with permissions, please try with:

```python
sudo python3 -m dashapp.app
```

You will be asked for a password. Insert it and continue.

### Author

Ana Isabel Casado Gomez