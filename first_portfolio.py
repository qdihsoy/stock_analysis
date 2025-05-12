from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import pandas as pd
import numpy as np

tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']
price_data = yf.download(tickers, start='2022-01-01', end='2024-12-31')['Adj Close']
returns = price_data.pct_change().dropna() # pct_change:勝手に日次リターンを計算してくれる

lagged = pd.concat([returns.shift(i) for i in range(1, 6)], axis=1)
lagged.columns = [f'{col}_lag{i}' for i in range(1, 6) for col in returns.columns]
X = lagged.dropna()
y = returns.loc[X.index]

model = LinearRegression()
model.fit(X, y)
y = returns.shift(-1).loc[X.index]