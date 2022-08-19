import datetime as dt
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional


def ML_stock(ticker: str):
    global company, scaled_data, prediction_days, x_train, y_train
    company = yf.Ticker(ticker)
    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2022, 1, 1)
    data = company.history(start=start, end=end)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 60
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # build the model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction of the next closing price
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=64)
    """Test model accuracy on existing data"""
    # load test data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    # test_data = web.Datareader(company, 'yahoo', test_start, test_end)
    test_data = company.history(start=test_start, end=test_end)  # period = "3mo", interval="1h")
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    # make prediction on test data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    # plot the test prediction
    plt.plot(actual_prices, color='black', label="actual prices")
    plt.plot(prediction_prices, color='green', label="predicted prices")
    plt.title(f"{company} prices")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.legend()
    plt.show()


ML_stock("CSCO")
