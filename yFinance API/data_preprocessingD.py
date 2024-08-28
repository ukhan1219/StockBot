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

# Download daily data for ES=F (E-mini S&P 500 Futures)
start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
end_date = datetime.now()
daily_data = download_data('ES=F', start_date, end_date, '1d')

# Save data to CSV
daily_data.to_csv('sp500_eminifutures_features_daily.csv')
