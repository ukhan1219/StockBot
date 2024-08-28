import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta

# Function to download data
def download_data(ticker, start_date, end_date, interval):
    data = pd.DataFrame()
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=59), end_date)
        chunk = yf.download(ticker, start=current_date.strftime('%Y-%m-%d'), end=next_date.strftime('%Y-%m-%d'), interval=interval)
        if not chunk.empty:
            data = pd.concat([data, chunk])
        current_date = next_date
    return data

# Load existing data
try:
    data = pd.read_csv('sp500_eminifutures_features_15m.csv', index_col='Date', parse_dates=True)
except FileNotFoundError:
    data = pd.DataFrame()

# Determine the start date for new data
if not data.empty:
    last_date = data.index[-1] + timedelta(minutes=15)  # Start from the next 15 minutes after the last date
else:
    last_date = datetime.now() - timedelta(days=59)

# Define the end date for the new data
end_date = datetime.now()

# Fetch 15-minute data for the last 59 days
minute_15_data = download_data('ES=F', last_date, end_date, '15m')
minute_15_data.index.name = 'Date'
minute_15_data.dropna(inplace=True)

# Combine with existing data
new_data = minute_15_data[~minute_15_data.index.duplicated(keep='last')]

# Create technical indicators as features
if not new_data.empty:
    new_data['SMA_50'] = ta.trend.sma_indicator(new_data['Close'], window=50)
    new_data['SMA_200'] = ta.trend.sma_indicator(new_data['Close'], window=200)
    new_data['RSI'] = ta.momentum.rsi(new_data['Close'], window=14)
    new_data.dropna(inplace=True)

# Append new data to existing data
updated_data = pd.concat([data, new_data])
updated_data = updated_data[~updated_data.index.duplicated(keep='last')]

# Save updated dataset
updated_data.to_csv('sp500_eminifutures_features_15m.csv')
