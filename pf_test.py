import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#データの取得
tickers = ['AAPL', 'AMZN', 'TSLA', '7203.T', '6758.T']

start_date = '2023-01-01'
end_date = '2023-12-31'

data = yf.download(tickers, start=start_date, end=end_date)

close_prices = data['Close'] #index:日付 columns:各銘柄の終値

# print(close_prices.head())

close_prices.to_csv('stock_prices.csv')

#特徴量作成
close_prices = pd.read_csv('stock_prices.csv', index_col=0, parse_dates=True) #日付列をdatetime型にしてインデックスへ

returns = close_prices.pct_change(fill_method=None).dropna() #欠損値を自動で埋めない

ma_5 = close_prices.rolling(window=5).mean()
ma_25 = close_prices.rolling(window=25).mean()

features = pd.concat([close_prices, returns, ma_5, ma_25], axis=1) #横に結合
features.columns = (
  [f'{col}_close' for col in close_prices.columns] +
  [f'{col}_return' for col in returns.columns] +
  [f'{col}_ma5' for col in ma_5.columns] +
  [f'{col}_ma25' for col in ma_25.columns]
  ) #リストの結合、カラム名変更

features = features.ffill().bfill() #欠損値を直前・直後の値で補完

features.to_csv('features.csv')

#ラグ特徴量の追加
lag_features = []

for lag in range(1, 6):
  lagged = returns.shift(lag)
  lagged.columns = [f'{col}_return_lag_{lag}' for col in returns.columns]
  lag_features.append(lagged)
  
df_lagged = pd.concat(lag_features, axis=1)
df_lagged = df_lagged.ffill().bfill()
df_lagged.to_csv('lag_features.csv')

lags = pd.read_csv('lag_features.csv', index_col=0, parse_dates=True)
features = pd.read_csv('features.csv', index_col=0, parse_dates=True)

X = pd.concat([features, lags], axis=1).ffill().bfill() #特徴量を一つのデータフレームに
y = returns['AAPL'].shift(-1) #翌日のリターンを正解データに

X, y = X.align(y, join='inner', axis=0)

