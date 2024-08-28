import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to download data
def download_data(ticker, start_date, end_date, interval):
    data = pd.DataFrame()
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=730), end_date)
        chunk = yf.download(ticker, start=current_date.strftime('%Y-%m-%d'), end=next_date.strftime('%Y-%m-%d'), interval=interval)
        if not chunk.empty:
            data = pd.concat([data, chunk])
        current_date = next_date
    return data

# Function to download and save data for different time frames
def save_data_for_intervals(ticker, intervals, end_date):
    for interval in intervals:
        if interval == '1h':
            start_date = end_date - timedelta(days=729)
        elif interval == '5m':
            start_date = end_date - timedelta(days=59)
        else:
            start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
        
        data = download_data(ticker, start_date, end_date, interval)
        filename = f'sp500_eminifutures_features_{interval}.csv'
        data.to_csv(filename)
        print(f"Saved data for interval {interval} to {filename}")

# Define time intervals you want to download
intervals = ['1d', '1h', '5m']  # Daily, Hourly, 5-Minute

# Set end date to current date
end_date = datetime.now()

# Download and save data for each interval
save_data_for_intervals('ES=F', intervals, end_date)
