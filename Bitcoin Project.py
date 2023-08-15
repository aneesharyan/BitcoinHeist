#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

data= "C:\\Users\\mishr\\OneDrive\\Desktop\\archive.zip"

# Creating lagged variables
lag_window = 5
for i in range(1, lag_window + 1):
    data[f'lag_{i}'] = data['value'].shift(i)

# Rolling statistics
rolling_window = 10
data['rolling_mean'] = data['value'].rolling(window=rolling_window).mean()

# Fourier transformations
def apply_fourier_transform(series, period):
    values = series.values
    n = len(values)
    time = np.arange(1, n + 1)
    freqs = np.fft.fftfreq(n)
    complex_result = np.fft.fft(values)
    amplitude = np.abs(complex_result)
    power = amplitude ** 2
    selected_freqs = freqs > 0  # Select positive frequencies
    selected_freqs[0] = False  # Exclude DC component
    selected_freqs = selected_freqs[:n // period]  # Keep up to the specified period
    amplitude = amplitude[selected_freqs]
    power = power[selected_freqs]
    return amplitude, power

fourier_period = 24  # Assuming daily periodicity
amplitude, power = apply_fourier_transform(data['value'], fourier_period)
data['fourier_amplitude'] = amplitude
data['fourier_power'] = power

# Split the data into training and validation sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]


# Anomaly Detection

# Combine features for anomaly detection
feature_cols = ['value', 'lag_1', 'rolling_mean', 'fourier_amplitude', 'fourier_power']
X_train = train_data[feature_cols]
X_val = val_data[feature_cols]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Apply One-Class SVM for anomaly detection
svm_model = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1)
svm_model.fit(X_train_scaled)
anomaly_scores = svm_model.decision_function(X_val_scaled)


# In[ ]:




# Split the data into training and validation sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# Individual Models

# ARIMA
def arima_forecast(train, val):
    history = [x for x in train]
    predictions = []
    for t in range(len(val)):
        model = ARIMA(history, order=(5,1,0))  # Example ARIMA order
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = val[t]
        history.append(obs)
    return predictions

arima_predictions = arima_forecast(train_data['value'], val_data['value'])

# LSTM
def lstm_forecast(train, val):
    train = np.array(train).reshape(-1, 1)
    val = np.array(val).reshape(-1, 1)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train, train, epochs=100, verbose=0)
    lstm_predictions = model.predict(val)
    return lstm_predictions

lstm_predictions = lstm_forecast(train_data['value'], val_data['value'])

# Prophet
def prophet_forecast(train, val):
    df = pd.DataFrame({'ds': train.index, 'y': train.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=len(val))
    prophet_forecast = model.predict(future)['yhat'][-len(val):].values
    return prophet_forecast

prophet_predictions = prophet_forecast(train_data['value'], val_data['value'])

# Ensemble

# Weights for the ensemble models (you can experiment with these)
arima_weight = 0.4
lstm_weight = 0.3
prophet_weight = 0.3

ensemble_predictions = (arima_weight * arima_predictions +
                        lstm_weight * lstm_predictions.flatten() +
                        prophet_weight * prophet_predictions)

# Calculate ensemble mean squared error
ensemble_mse = mean_squared_error(val_data['value'], ensemble_predictions)
print(f"Ensemble MSE: {ensemble_mse}")



# In[ ]:




