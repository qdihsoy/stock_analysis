import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

ticker = '^DJI' #ティッカーシンボル:銘柄コード
start_date = "2023-01-01"
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
  print('データが取得できませんでした。ティッカーや日付を確認してください。')
else:
  plt.figure(figsize=(10, 5))
  plt.plot(data['Close'], label='Close Price')
  plt.title('DJI Close Price')
  plt.xlabel('Date')
  plt.ylabel('Price(JPY)')
  plt.legend() #凡例の表示
  plt.grid(True)

  close_prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close']['^DJI']

  mean_price = close_prices.mean()
  max_price = close_prices.max()
  min_price = close_prices.min()
  std_dev = close_prices.std()
  count = close_prices.count()

  print('日経平均株価（終値）の統計情報')
  print(f'データ件数：{count}')
  print(f'平均価格：{mean_price:.2f}円') #.2f:小数点以下２桁で表示
  print(f'最大価格：{max_price:.2f}円')
  print(f'最小価格：{min_price:.2f}円')
  print(f'標準偏差：{std_dev:.2f}円')

ma_7 = close_prices.rolling(window=7).mean()
ma_30 = close_prices.rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='Close Price', color='blue')
plt.plot(ma_7, label='7-day MA', color='orange')
plt.plot(ma_30, label='30-day MA', color='green')
plt.title('DJI Close Price with Moveng Averages')
plt.xlabel('Date')
plt.ylabel('Price(JPY)')
plt.legend()
plt.grid(True)
plt.tight_layout()

df = data[['Close']].copy() #ダブル[]にすることでCloseだけを抜き出して新しデータフレームを作る/.copy:もとのdataに影響を与えないようにコピーしている
df.dropna(inplace=True) #欠損値を含む行を削除/inplace=Trueで書き換え

df['MA7'] = df['Close'].rolling(window=7).mean() #rolling:移動しながらwindow:計算する期間
df['MA30'] = df['Close'].rolling(window=30).mean()

df.dropna(inplace=True) #7日移動平均や30日移動平均は、最初の数行は平均が計算できず NaN になるため、それらを再び削除

print(df.head())

from sklearn.linear_model import LinearRegression #回帰モデル
from sklearn.model_selection import train_test_split #データを「訓練用」と「テスト用」に分けるための関数

X = df[['MA7', 'MA30']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #20％をテストデータに、時系列データなのでシャッフルせず分解

model = LinearRegression()
model.fit(X_train, y_train) #残り80%のデータを渡し、学習させる

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Stock Price Prediction (Linear Regression)')
plt.xlabel('Time')
plt.ylabel('Price (JPY)')
plt.legend()
plt.grid(True)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\n モデルの評価指標')
print(f'平均二乗誤差（MSE):{mse:.2f}')
print(f'平方平均二乗誤差（RMSE）:{rmse:.2f}')
print(f'平均絶対誤差（MAE）:{mae:.2f}')
print(f'決定係数 R²:{r2:.3f}')

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual', color='blue') #y_test.index:日付のインデックス
plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.legend()
plt.grid(True)

data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_25'] = data['Close'].rolling(window=25).mean()

data.dropna(inplace=True)

X = data[['MA_5', 'MA_25']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

predicted = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, predicted))

print(f'RMSE（二乗平均平方根誤差）:{rmse:.2f}円')

plt.figure(figsize=(12,6))
plt.plot(data.index, y, label='Actual Close Price')
plt.plot(data.index, predicted, label='Predicted Close Price (with MA)', linestyle='--')
plt.title('DJI Prediction using Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.legend()
plt.grid(True)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
rf_predicted = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predicted))

print(f'ランダムフォレストのRMSE:{rf_rmse:.2f}円')

data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)

data = data.dropna()

features = data[['Close_lag1', 'Close_lag2']]
target = data['Close']

X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE:{rmse:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Price (JPY)')
plt.grid(True)

plt.show()