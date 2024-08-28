import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import joblib
import os
import ta
import shutil

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Clear previous tuner directory
shutil.rmtree('my_dir/intro_to_kt', ignore_errors=True)

# Load data
interval = '15m'  # Change this to switch between different intervals ('1d', '1h', '5m', '15m')
index_col = 'datetime' if interval in ['1h', '5m', '15m'] else 'Date'

# Read the CSV file
data = pd.read_csv(f'sp500_eminifutures_features_{interval}.csv', index_col=index_col, parse_dates=True)

# Debugging: Check the dataframe columns
print(f"Data Columns: {data.columns}")

# Add time-related features if using smaller intervals
if interval in ['1h', '5m', '15m']:
    data['Hour'] = data.index.hour
    data['Minute'] = data.index.minute

# Add technical indicators
data['SMA_50'] = data['close'].rolling(window=50).mean()
data['SMA_200'] = data['close'].rolling(window=200).mean()
data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
data['MACD'] = ta.trend.MACD(data['close']).macd()
data['Bollinger_Upper'] = ta.volatility.BollingerBands(data['close']).bollinger_hband()
data['Bollinger_Lower'] = ta.volatility.BollingerBands(data['close']).bollinger_lband()

# Drop rows with NaN values
data.dropna(inplace=True)

# Define features and target
feature_cols = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
if interval in ['1h', '5m', '15m']:
    feature_cols.extend(['Hour', 'Minute'])

X = data[feature_cols].values
y = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 if next price is higher, else 0

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the tuner
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# Perform the hyperparameter search
tuner.search(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters for debugging
print("Best hyperparameters:")
for key, value in best_hyperparameters.values.items():
    print(f"{key}: {value}")

# Train the best model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = best_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Save the trained model in the recommended format
best_model.save('trading_model.keras')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Print the training history for debugging purposes
print("Training history:", history.history)
