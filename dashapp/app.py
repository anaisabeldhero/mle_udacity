from datetime import date, timedelta, datetime

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dashapp.helping_functions.dash_dax30 import get_data, predict_next_days
from utils.data.data_config import LIST_DAX_COMPANIES

# Initialise the app
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the app
app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                   html.Div(className='four columns div-user-controls',
                                            children=[
                                                html.A(html.Img(src='/assets/dax30_logo.jpeg', className='logo'),
                                                       href='/'),
                                                html.H1('STOCK PREDICTOR PRICES', style={'font-weight': 'bold'}),
                                                html.H4("Next best Action for your current or future investment."),
                                                html.P("Please follow the next steps carefully..."),
                                                html.Br(),
                                                html.I("1. Pick one stock from the dropdown below."),
                                                html.Div(
                                                    className='div-for-dropdown',
                                                    children=[
                                                        dcc.Dropdown(id='stockselector',
                                                                     options=[{'label': i,
                                                                               'value': i} for i in LIST_DAX_COMPANIES],
                                                                     multi=False,
                                                                     style={'backgroundColor': '#1E1E1E'},
                                                                     placeholder="Select a symbol",
                                                                     className='stockselector'
                                                                     ),
                                                    ],
                                                    style={'color': '#1E1E1E'}),
                                                html.I("2. Do you have already stocks invested in this company?\n"),
                                                html.Div(
                                                    className='stock-acquisition-confirmation',
                                                    children=[
                                                        dcc.RadioItems(options=[{'label': 'Yes', 'value': 1},
                                                                                {'label': 'No', 'value': 0}],
                                                                       # value=0,
                                                                       id="own_shares",
                                                                       labelStyle={'display': 'inline-block'})
                                                    ]),
                                                html.Br(),
                                                html.I("3. If so, how many of them? Please introduce the amount."),
                                                html.Div([
                                                        dcc.Input(id="stock_amount", type="number", placeholder="1"),
                                                        html.Div(id="output")]
                                                ),
                                                html.Br(),
                                                html.I("4. When did you acquire them? Please introduce the date."),
                                                html.Div([
                                                    dcc.DatePickerSingle(
                                                        id='my-date-picker-single',
                                                        min_date_allowed=date(2016, 1, 1),
                                                        max_date_allowed=date.today(),
                                                        initial_visible_month=date.today(),
                                                        style={'backgroundColor': '#cb4fdb'},
                                                    ),
                                                    html.Div(id='output-container-date-picker-single')
                                                ]),
                                                html.Br(),
                                                html.P("When you are ready, click 'Ready' button!"),
                                                html.Div([
                                                    html.Button('Ready!', id='ready-button', n_clicks=0),
                                                ])

                                            ]),  # Define the left element
                                   html.Div(className='eight columns div-for-charts bg-grey',
                                            children=[dcc.Graph(id='timeseries', config={'displayModeBar': False},
                                                                animate=False),
                                                      html.P("Tip: Zoom in the predictions to see it closely."),
                                                      html.Div(id='current_investment'),
                                                      html.Div(id='gain_text'),
                                                      html.H5("1. Decision about your investment...",
                                                              style={'font-weight': 'small-caps'}),
                                                      html.P(id='decision_investment'),
                                                      html.H5("2. Decision about the company...",
                                                              style={'font-weight': 'small-caps'}),
                                                      html.P(id='decision_company'),
                                                      ]
                                            )  # Define the right element
                                   ])
                      ])


@app.callback(
    [
        Output('timeseries', 'figure'),
        Output('gain_text', 'children'),
        Output('decision_investment', 'children'),
        Output('decision_company', 'children'),
        Output('current_investment', 'children')
    ],
    [
        Input('stockselector', 'value'),
        Input('own_shares', 'value'),
        Input('stock_amount', 'value'),
        Input('my-date-picker-single', 'date'),
        Input('ready-button', 'n_clicks')
    ]
)
def update_graph(stockselector_symbol, own_shares, stock_amount, date_stock, n_clicks):

    if n_clicks > 0:
        trace_series, trace_pred = [], []
        df = get_data(stockselector_symbol)

        predictions, decision_investment, decision_company, investment = predict_next_days(df, own_shares,
                                                                                           stock_amount,
                                                                                           date_stock)

        df_expand = df.reindex(pd.date_range(min(df['Adj Close'].index),
                                             max([datetime.today() + timedelta(days=i) for i in range(5)])).tolist())

        for stock in [stockselector_symbol]:
            trace_series.append(go.Scatter(x=df_expand['Adj Close'].index,
                                           y=[np.nan] * len(df) + predictions.tolist(), mode='markers', opacity=0.7,
                                           name='predictions',
                                           textposition='bottom center'))

            trace_series.append(go.Scatter(x=df_expand['Adj Close'].index,
                                           y=df['Adj Close'].tolist() + [np.nan] * 5,
                                           mode='lines', opacity=0.7,
                                           name=stock,
                                           textposition='bottom center'))

        data_series = [val for sublist in [trace_series] for val in sublist]

        figure1 = {'data': data_series,
                   'layout': go.Layout(
                       colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                       template='plotly_dark',
                       paper_bgcolor='rgba(0, 0, 0, 0)',
                       plot_bgcolor='rgba(0, 0, 0, 0)',
                       margin={'b': 15},
                       hovermode='x',
                       autosize=True,
                       yaxis_title="Euro - €",
                       title={'text': f'{stockselector_symbol} Stock Prices & Predictions next 5 days',
                              'font': {'color': '#5E0DAC'}, 'x': 0.5},
                       xaxis={'range': [df_expand.index.min(), df_expand.index.max()]}
                    )}

        gain_forecast = round(predictions[4]-predictions[0], 2)
        text_gain = f"Possible maximum increase in the next 5 days: {gain_forecast}€."

        return [figure1,
                html.H4([dbc.Alert(text_gain, color='info')]),
                html.H4([dbc.Alert(decision_investment[0] if own_shares
                                   else "You don't own any shares. We can't advice here.",
                                   color=decision_investment[1] if own_shares else 'primary')]),
                html.H4([dbc.Alert(decision_company[0], color=decision_company[1])]),
                html.H4(f" The status of your investment is {round(investment[0], 2)}€, "
                        f"initial was {round(investment[1], 2)}€, "
                        f"the difference is {round(investment[0] - investment[1], 2)}€.",
                        style={'color': '#de583d'})
                ]
    else:
        raise PreventUpdate


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
