from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
# Initialize the tvDatafeed object with authentication
username = USERNAME
password = PASSWORD
tv = TvDatafeed(username, password)

# Download data function
def download_data(ticker, exchange, interval, n_bars):
    data = tv.get_hist(ticker, exchange, interval, n_bars=15000)
    return data

# Download 15-minute data for ES=F (E-mini S&P 500 Futures)
ticker = 'ES1!'  # S&P 500 E-mini Futures
exchange = 'CME_MINI'
interval = Interval.in_15_minute
n_bars = 500000  # Adjust based on maximum allowed by API

# Get the data
data = download_data(ticker, exchange, interval, n_bars=n_bars)

# Save data to CSV
data.to_csv('sp500_eminifutures_features_15m.csv')

print("Data saved successfully!")
