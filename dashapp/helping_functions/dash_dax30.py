"""
This script contains the function used in the app.py in order to download the data, predict the next 5 days
and take the best next action decision.
"""
from datetime import timedelta, datetime
import yfinance as yf

from utils.data_preprocessing import impute_data_points
from utils.models_class import ARIMAModel


def get_data(symbol):
    """
    Function to download the data for a symbol and process the data set.
    :param symbol: company symbol to download the data.
    :return: the preprocessed dataset
    """
    symbol_df = yf.download(symbol, start="2016-01-01", end=datetime.today())
    symbol_df = symbol_df.asfreq('D')
    symbol_df_imp = impute_data_points(symbol_df)
    return symbol_df_imp


def predict_next_days(df, own_shares, number_stocks, date_stocks):
    """
    This function holds the model training, predict and the return computation to feed the decision functions.
    :param df: preprocessed dataframe
    :param own_shares: boolean to know if the client already has stocks for the current symbol or not.
    :param number_stocks: the number of symbols the client own
    :param date_stocks: the date when the client acquired the stocks
    :return: predictions, decision for investment, decision about the company, list with current investment and initial
    """
    arima_model = ARIMAModel(6, 1, 0, 5)  # p, d, q, prediction horizon
    arima_model.prepare_data_train(df, 'Adj Close')
    arima_model.fit_predict()

    df['Returns'] = df['Adj Close'] / df['Adj Close'].shift(1)  # return for all the sample

    if own_shares:
        # parameters from user
        number_of_stocks = number_stocks
        date_stock_acquisition = date_stocks
    else:
        # if the user hasn't invested then we will use fake parameters.
        number_of_stocks = 100
        date_stock_acquisition = datetime.today() - timedelta(days=28)

    # how much the user can gain in the next 5 days?
    gain_forecast = round(arima_model.predictions[4] - arima_model.predictions[0], 2)

    # computation of your current return and position
    df_since_stocks_acq = df.loc[date_stock_acquisition:]
    df_since_stocks_acq['Norm return'] = df_since_stocks_acq['Adj Close'] / df_since_stocks_acq.iloc[0]['Adj Close']
    df_since_stocks_acq['Position'] = df_since_stocks_acq['Norm return'] * number_of_stocks
    df_since_stocks_acq['Daily Return'] = df_since_stocks_acq['Position'].pct_change()  # default is 1

    status_inversion = df_since_stocks_acq['Position'][-1] * df_since_stocks_acq['Adj Close'][0] if own_shares else 0
    investment = df_since_stocks_acq['Adj Close'][0] * number_of_stocks

    # we will base the decision function in the normed return gained description
    norm_return_description = df_since_stocks_acq['Norm return'].describe()

    # definition of moderate strategy, 10% of loses and 30% on gains.
    game_range = [investment - investment * 0.1, investment + investment * 0.3]
    forecast = df_since_stocks_acq['Adj Close'][0] * df_since_stocks_acq['Position'][-1] + (
                gain_forecast * number_of_stocks)

    decision_next_investment = decision_best_next_investment(forecast, game_range)
    decision_company = decision_best_next_company(norm_return_description)

    return arima_model.predictions, decision_next_investment, decision_company, [status_inversion,
                                                                                 investment if own_shares else 0]


def decision_best_next_investment(forecast, game_range):
    """
    Function to advice about the best next action with your investment. It's based on the moderate strategy of
    don't lose more than 10% of investment and sell after 30% of gaining in the investment.
    :param forecast: the forecast  amount.
    :param game_range: list with thresholds of 10% losses and 30% gainings that we are willing to accept.
    :return: list of decision with text and color of alert.
    """
    if forecast < game_range[0]:
        return [f"Decision about your investment... ++ Sell now or you will lose more than 10% of your inversion.",
                "danger"]
    elif forecast > game_range[1]:
        return [f"Decision about your investment... - Sell now to gain more than 20% of your investment!", "success"]
    else:
        return [f"Decision about your investment... + Hold On, game is still up.", "warning"]


def decision_best_next_company(norm_return_description):
    """
    Function to advice about the current status of the company based on your investment or in the last 14 days
    performance. The Thresholds are decided in the context of the mean of the return.
    As it's normed it should be close to 1. With the moderate strategy followed here, 0.15 over mean will be ~30% gains.
    Less than 1 won't be interested for us.
    :param norm_return_description: stats of the norm return variable
    :return: list of decision with text to show as decision in the dashboard and the color of alert.
    """
    if norm_return_description['mean'] >= 1.15:
        return ["Decision about the company... ++ Usually this company wins, invest more!", "success"]
    elif 1.01 < norm_return_description['mean'] < 1.15:
        return ["Decision about the company... + This company is promising! Invest if you haven't!", "success"]
    elif norm_return_description['mean'] < 0.95:
        return ["Decision about the company... - Usually this company performs low.", "danger"]
    else:
        return ["Decision about the company... - This company don't have enough volatility, right now.", "warning"]
