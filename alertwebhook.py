from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    action = data.get('action', 'No action received')
    print(f"Received action: {action}")
    
    # Send the action to TradingView via a custom service or automation tool
    send_to_tradingview(action)
    
    return jsonify({"status": "success", "action_received": action}), 200

def send_to_tradingview(action):
    tradingview_url = 'https://your-automation-service-url'  # Replace with your automation service URL
    headers = {'Content-Type': 'application/json'}
    payload = {
        "action": action,
        "message": f"Alert: {action}"
    }
    response = requests.post(tradingview_url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Alert created successfully on TradingView")
    else:
        print("Failed to create alert on TradingView")

if __name__ == '__main__':
    app.run(port=5000)
