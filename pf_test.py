import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['AAPL', 'AMZN', 'TSLA', '7203.T', '6758.T']

start_date = '2023-01-01'
end_date = '2023-12-31'

data = yf.download(tickers, start=start_date, end=end_date)

close_prices = data['Close']

print(close_prices.head())

close_prices.to_csv('stock_prices.csv')