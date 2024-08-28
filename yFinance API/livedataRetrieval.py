import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests

# Load the trained model and scaler
model = load_model('trading_model.keras')
scaler = joblib.load('scaler.pkl')

# Function to get live price data
def get_live_data(symbol, interval):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=30)  # Get the last 30 minutes data
    data = yf.download(symbol, start=start_time, end=end_time, interval=interval)
    if data.empty:
        print(f"No data retrieved for {symbol} between {start_time} and {end_time}")
    return data

# Function to preprocess data
def preprocess_data(data):
    # Assuming data is a DataFrame with columns similar to the training data
    data = data.values.reshape(-1, 1)  # Reshape if necessary
    scaled_data = scaler.transform(data)
    return scaled_data

# Function to make predictions
def make_prediction(data):
    prediction = model.predict(data)
    return np.argmax(prediction)  # Assuming classification model with buy/sell/hold

# Function to output recommendations
def output_recommendation(action, timestamp):
    actions = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
    print(f"{timestamp}: Recommendation - {actions[action]}")
    return actions[action]

# Function to send alerts to the Flask webhook
def send_webhook_alert(action):
    url = 'http://localhost:5000/webhook'  # Replace with your server's IP or domain
    message = {"action": action}
    response = requests.post(url, json=message)
    if response.status_code == 200:
        print(f"Webhook alert sent: {action}")
    else:
        print(f"Failed to send webhook alert: {action}")

# Main loop
symbol = 'ES=F'  # S&P 500 E-mini Futures
interval = '5m'  # 5-minute candles

while True:
    # Get live data
    data = get_live_data(symbol, interval)
    if not data.empty:
        # Preprocess data
        processed_data = preprocess_data(data)
        # Make prediction
        action = make_prediction(processed_data)
        # Output recommendation
        action_str = output_recommendation(action, datetime.utcnow())
        # Send webhook alert
        send_webhook_alert(action_str)
    
    # Wait for the next candle
    time.sleep(60)  # 5 minutes
