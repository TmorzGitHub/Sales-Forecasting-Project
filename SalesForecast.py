# # Sales Forecasting Project - Step-by-Step Script

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # Step 1: Load the dataset
file_path = 'C:\Documents\DataAnalyticsPortfolio\SalesForecastingModel\superstoredata.csv'

# # Load data
df = pd.read_csv(file_path)

# # Step 2: Preprocess the data
df['Order Date'] = pd.to_datetime(df['Order Date']) # Convert to datetime
df = df.set_index('Order Date') # Set as index
sales = df['Sales'].resample('D').sum() # Aggregate daily sales

# # Check for missing values
missing_values = sales.isnull().sum()
if missing_values > 0:
    print("Missing values found:", missing_values)
else:
    print("No missing values found.")

# # Step 3: Plot daily sales over time
# plt.figure(figsize=(12,6))
# sales.plot()
# plt.title('Daily Sales Over Time')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.show()

# # Step 4: Perform ADF test to check stationarity
result = adfuller(sales.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] <= 0.05:
    print("Series is likely stationary.")
else:
    print("Series may need differencing.")

# # Step 5: Fit SARIMA model
sarima_model = SARIMAX(sales,
                       order=(1,1,1),
                       seasonal_order=(1,1,1,7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
results = sarima_model.fit()

# # Step 6: Forecast future sales
forecast = results.get_forecast(steps=30)
forecast_index = pd.date_range(start=sales.index[-1], periods=31, freq='D')[1:]
forecast_values = forecast.predicted_mean

plt.figure(figsize=(12,6))
plt.plot(sales, label='Historical Sales')
plt.plot(forecast_index, forecast_values, color='red', label='Forecasted Sales')
plt.title('Sales Forecast with SARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# # Step 7: Analyze residuals
residuals = sales - results.fittedvalues

# plt.figure(figsize=(12,6))
# plt.scatter(results.fittedvalues, residuals)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residuals of SARIMA Model')
# plt.xlabel('Fitted Values')
# plt.ylabel('Residuals')
# plt.show()

# # Step 8: Plot ACF and PACF of residuals
# plt.figure(figsize=(12,6))
# plot_acf(residuals.dropna(), lags=50)
# plt.title('ACF of Residuals')
# plt.show()

# plt.figure(figsize=(12,6))
# plot_pacf(residuals.dropna(), lags=50)
# plt.title('PACF of Residuals')
# plt.show()

# # Instructions for Running the Script:
# # 1. Ensure the dataset file is present in the specified path.
# # 2. Run the script as is for the forecast output
# # 3. Uncomment and run the script sections as needed for model evaluation