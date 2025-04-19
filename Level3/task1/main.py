import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Step 1: Create a time series (simulate daily sales data)
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
sales = 200 + np.sin(np.linspace(0, 50, len(date_rng))) * 20 + np.random.normal(0, 5, len(date_rng))  # trend + seasonality + noise
df = pd.DataFrame({'date': date_rng, 'sales': sales})
df.set_index('date', inplace=True)

# Step 2: Plot and decompose time series
decomposition = seasonal_decompose(df['sales'], model='additive', period=365)
decomposition.plot()
plt.suptitle("Time Series Decomposition", fontsize=16)
plt.tight_layout()
plt.show()

# Step 3: Moving Average
df['MA_30'] = df['sales'].rolling(window=30).mean()

# Step 4: Exponential Smoothing
es_model = ExponentialSmoothing(df['sales'], seasonal='add', seasonal_periods=365).fit()
df['ES'] = es_model.fittedvalues

# Plot original vs smoothed
plt.figure(figsize=(10, 5))
plt.plot(df['sales'], label='Original')
plt.plot(df['MA_30'], label='30-day Moving Average', color='orange')
plt.plot(df['ES'], label='Exponential Smoothing', color='green')
plt.legend()
plt.title("Smoothed Time Series")
plt.grid()
plt.show()

# Step 5: Build ARIMA model
train = df['sales'][:-90]
test = df['sales'][-90:]

arima_model = ARIMA(train, order=(5, 1, 2))  # p=5, d=1, q=2 (these can be tuned)
arima_result = arima_model.fit()

forecast = arima_result.forecast(steps=90)
rmse = sqrt(mean_squared_error(test, forecast))
print(f"📈 ARIMA Model RMSE: {rmse:.2f}")

# Step 6: Plot forecast vs actual
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red')
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.grid()
plt.show()
