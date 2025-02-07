import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import optuna
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def compute_technical_indicators(df):
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().rolling(14).mean() / df['Close'].diff().rolling(14).std()))
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
    return df.dropna()

def compute_advanced_features(df):
    df['Momentum'] = df['Close'].pct_change(periods=10)
    df['ADX'] = df['High'].diff(1).abs().rolling(14).mean() / df['Close'].rolling(14).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['MA_Short'] = df['Close'].rolling(window=10).mean()
    df['MA_Long'] = df['Close'].rolling(window=50).mean()
    df['MA_Ratio'] = df['MA_Short'] / df['MA_Long']
    return df.dropna()

def prepare_data(df, sequence_length, selected_features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[selected_features])
    scaler_close = MinMaxScaler()
    df['Close_Scaled'] = scaler_close.fit_transform(df[['Close']])  
    X, y = [], []
    for i in range(sequence_length, len(df_scaled) - 1):
        X.append(df_scaled[i - sequence_length:i])
        y.append(df_scaled[i, 0])  
    return np.array(X), np.array(y), scaler, scaler_close

def build_lstm_model(input_shape, dropout_rate):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(100, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape, dropout_rate):
    model = Sequential([
        GRU(100, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        GRU(100, return_sequences=True),
        Dropout(dropout_rate),
        GRU(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler_close):
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    predictions = model.predict(X_test)
    predictions_actual = scaler_close.inverse_transform(predictions)
    y_test_actual = scaler_close.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(np.mean((predictions_actual - y_test_actual) ** 2))
    return rmse, predictions_actual, y_test_actual


def optimize_model(model_type):
    def objective(trial):
        sequence_length = trial.suggest_int('sequence_length', 10, 30, step=5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.05)
        available_features = ['Close', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'Momentum', 'ADX', 'Volume_Change', 'MA_Ratio']
        selected_features = ['Close'] + [feature for feature in available_features if trial.suggest_categorical(f"use_{feature}", [True, False])]
        if len(selected_features) < 3:
            return float('inf')
        X, y, scaler, scaler_close = prepare_data(df, sequence_length, selected_features)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), dropout_rate) if model_type == 'LSTM' else build_gru_model((X_train.shape[1], X_train.shape[2]), dropout_rate)
        rmse, predictions_actual, y_test_actual = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler_close)
        return rmse, predictions_actual, y_test_actual
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    best_selected_features = ['Close'] + [feature for feature in ['RSI', 'MACD', 'Lower_Band', 'Volume_Change', 'Momentum', 'ADX', 'MA_Ratio'] if best_params.get(f'use_{feature}', True)]
    best_params['selected_features'] = best_selected_features
    best_params['rmse'] = study.best_value
    return best_params, predictions_actual, y_test_actual


ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
df = fetch_stock_data(ticker, start_date, end_date)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0) 
df = compute_technical_indicators(df)
df = compute_advanced_features(df)

lstm_params, predictions_actual, y_test_actual = optimize_model('LSTM')
gru_params, predictions_actual, y_test_actual = optimize_model('GRU')

best_model, best_rmse = min(
    [('LSTM', lstm_params['rmse']), ('GRU', gru_params['rmse'])],
    key=lambda x: x[1]
)

print(f'🔍 Best Model: {best_model} with RMSE: {best_rmse}')

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(predictions_actual, label='Predicted Prices', color='red', linestyle='dashed')
plt.title(f'Stock Price Prediction with {best_model} (RMSE: {best_rmse:.2f})')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()