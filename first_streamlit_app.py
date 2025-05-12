import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

st.title("株価予測ダッシュボード")

ticker = st.text_input("ティッカーを入力してください（例：^N225, AAPL, MSFT）", "^DJI")
start_date = st.date_input("開始日", pd.to_datetime("2023-01-01"))
end_date = st.date_input("終了日", pd.to_datetime("2024-12-31"))

if st.button("予測を実行"):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("データが取得できませんでした。")
    else:
        st.success(f"{ticker} のデータを取得しました。")

        data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20日移動平均
        data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50日移動平均
        data['Lag_1'] = data['Close'].shift(1)  # ラグ特徴量（前日終値）

        features = ['SMA_20', 'SMA_50', 'Lag_1']
        target = 'Close'

        data = data.dropna()

        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.write(f"RMSE: {rmse}")
        st.write(f"R²: {r2}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test.index, y_test, label="実際の株価")
        ax.plot(y_test.index, y_pred, label="予測株価", linestyle='--')
        ax.set_xlabel("日付")
        ax.set_ylabel("株価")
        ax.set_title(f"{ticker} prediction")
        ax.legend()
        st.pyplot(fig)