from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trading_model.keras')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json

    # Extract features from the webhook data
    features = np.array([[data['SMA_50'], data['SMA_200'], data['RSI'], data['MACD'], data['Bollinger_Upper'], data['Bollinger_Lower']]])
    features_scaled = scaler.transform(features)

    # Make predictions
    predictions = model.predict(features_scaled)[0]

    # Determine trade actions
    action = 'buy' if predictions[0] > 0.5 else 'sell'
    take_profit = 'yes' if predictions[1] > 0.5 else 'no'
    stop_loss = 'yes' if predictions[2] > 0.5 else 'no'

    # Return trade actions
    return jsonify({'action': action, 'take_profit': take_profit, 'stop_loss': stop_loss})

if __name__ == '__main__':
    app.run(port=80)
