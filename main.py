import os
import random
from datetime import date
import datetime as dt
import math
import cmath
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

from tensorboard.plugins.hparams import api as hp
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
# from yfinance import stock_info as si
from collections import deque
import yfinance as yf
import scipy.fft
import matplotlib.pyplot as plotter
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import numpy as np
import scipy.fftpack
from datetime import date
from scipy import signal
from collections import Counter
import statistics as stat

# for testing pushes - delete after
#TODO - set the values to go from specific dates and not all the data dates (from 2012)
# address the problem that the recommendations are a set greater then 5.
def score_of_stock_based_on_recommendations(company_tag):
    dictionary = {}
    i = 0
    while (i < company_tag.recommendations["To Grade"].count()):
        rec = company_tag.recommendations["To Grade"][i]
        if (dictionary.get(rec) == None):
            dictionary[rec] = 1
        else:
            dictionary[rec] += 1
        i += 1
    score = 0
    for rec in dictionary:
        number_of_item_recorded = dictionary[rec]
        if (rec == "Hold" or rec == "Positive"):
            score += number_of_item_recorded * 3
        if (rec == "Buy" or rec == "Long-Term Buy"):
            score += number_of_item_recorded * 10
        if (
                rec == "Overweight" or rec == "Outperform" or rec == "Strong Buy" or rec == "Market Outperform" or rec == "Sector Outperform"):
            score += number_of_item_recorded * 12
        if (
                rec == "Neutral" or rec == "Perform" or rec == "Market Perform" or rec == "Equal-weight" or rec == "Equal-Weight"):
            score += number_of_item_recorded * 0
        if (rec == "Sell"):
            score += number_of_item_recorded * -10
    average_score = score / company_tag.recommendations["To Grade"].count()
    print("Based on recommendations of analysts - the total score is:", average_score)
    print(dictionary)


def mean_of_func (arr: list):
    return sum(arr)/len(arr)

def abs_mean_of_func (arr: list):
    return abs(sum(arr)/len(arr))

def fetch_plot_close_column(ticker_name: str):
    company_tag = yf.Ticker(ticker_name)

    today = date.today()
    history_data = company_tag.history(period="1d",interval="1m")
    # history_data = company_tag.history(start="2022-07-22",  end=today, interval="1m")
    history_data['Close'].plot(title=f"{company_tag}stock price ($)")
    plt.show()
    return(history_data)


def SLTM_ML(company_tag: str):
    # Load data
    company = yf.Ticker(company_tag)

    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2022, 1, 1)

    data = company.history(start=start, end=end)

    # Prepare data
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

    model.add(LSTM(units=300, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=300, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
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
    # WE NEED TO RETURN IF THE STOCK GOES UP DOWN.


#TODO nitzan
def find_phys_elements(arr):

    original_value_quantity = len(arr)
    print(f"the number of values obtained: {original_value_quantity}")

    arr_mean = sum(arr) / len(arr)
    print(f"The mean of all values of the given function PLUS zeros at the end is: {arr_mean}")

    #List of physics values we calculate:

    Delta_x_arr = np.zeros(len(arr))
    Velocity = np.zeros(len(arr))
    Delta_v = np.zeros(len(arr))
    Accileration = np.zeros(len(arr))
    Energy = np.zeros(len(arr))
    Delta_E = np.zeros(len(arr))
    Momentum = np.zeros(len(arr))

    #calculation of values:

    print("HERE COMES YYYYYYYY - STEP")
    i = 0
    while i < len(arr) - 1:
        Delta_x_arr[i] = arr[i + 1] - arr[i]
        i += 1
    Delta_x_arr[i] = Delta_x_arr[i - 1]                         # The array formed out of gaps is one item short - hence we complete it.


    Delta_x_arr_Mean = sum(Delta_x_arr) / len(Delta_x_arr)
    abs_Delta_x_arr_Mean = sum(abs(Delta_x_arr)) / len(Delta_x_arr)
    print(f"\n\nSTEP MEAN IS: {Delta_x_arr_Mean} and the abs_STEP_MEAN IS: {abs_Delta_x_arr_Mean} \n\n")

    m = abs_Delta_x_arr_Mean

    print("HERE COMES THE SECOND YYYYYYY - SPEED & Energy")
    i = 0
    while i < len(arr) - 1:
        Velocity[i] = Delta_x_arr[i]                # because V = Delta_x / t, and t is const so doesn't matter.
        Delta_v[i] = Delta_x_arr[i + 1] - Delta_x_arr[i]
        Accileration[i] = Delta_v[i]                # because A = Delta_v / t, and t is const so doesn't matter.
        Energy[i] = 0.5 * m * (abs(Delta_x_arr[i])**2)
        Delta_E[i] = m * Velocity * Delta_v
        Momentum[i] = m * Velocity
        i += 1

    Velocity[i] = Delta_x_arr[i]
    Delta_v[i] = Delta_v[i - 1]
    Accileration[i] = Accileration[i-1]
    Energy[i] = Energy[i-1]
    Delta_E[i] = Delta_E[i-1]
    Momentum[i] = Momentum[i-1]

    #printing the phys values:

    t = np.arange(0, len(Delta_x_arr))
    plt.plot(t, Delta_x_arr)
    plt.title("Delta_x function")
    plt.ylabel("Delta_x")
    plt.xlabel("time")
    plt.show()


########################################
# End of TODO Nitzan

#TODO Nitzan
def extend_signal(signal):
    signal_multi_num = 1
    new_signal = np.hstack([signal] * signal_multi_num)  # the actual concat - everytime it grows with 1
    """end of concat"""

    zeros_to_concat = 10
    number_of_zeros = zeros_to_concat * (len(signal)-1)

    i = 0
    while i < number_of_zeros:  # inserting zeros at the end of y_high function vector to maitain a better freq picture.
        new_signal = np.append(signal, 0)
        i = i + 1

    signal_multi_growth = number_of_zeros * signal_multi_num

    return (new_signal,signal_multi_growth)

####################################################################
#TODO
def plot_fft(signal,signal_multi_growth):
    Fs = float(len(signal))  # range of sampling from 0
    Ts = 1.0  # what will be the sampling rate when considering THE FUCKING FREQUENCIES OF THE SYSTEM.
    t = np.arange(0, Fs, Ts)  # time vector

    y = signal.copy()
    plt.plot(t, y)
    # plt.show() //

    n = len(y)  # length of the signal we created = Fs/Ts
    k = np.arange(n)  # basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
    T = n * Ts  # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
    frq = k / T  # two sides frequency range
    frq = frq[:len(frq) // 2]  # one side frequency range
    Y = np.fft.fft(y)  # dft and normalization
    Y_half = Y[:n // 2]
    a = np.angle(Y)
    a_half = a[:n // 2]

    y_new = 0
    The_length_I_want_to_see = Fs  ################ DO THE SAME IN THE TESTING SIGNAL TO CHECK IT OUT!!!!!!!
    steps = The_length_I_want_to_see / len(y)
    x = np.arange(0, The_length_I_want_to_see,steps)  # BUILDING THE EXACT SAME VECTOR THAT WAS FOR THE ORIGINAL SIGNAL - t....
    maximum = max(abs(Y_half))

    i = 0
    while (i < len(frq)):
        if (abs(Y_half[i]) > 0.75 * maximum):  # whoever answers to the conditions will be represented in the recreated function
            # if (i < 16):
            print(f"we have frequency {i / (Fs * signal_multi_growth)} with the value of {abs(Y_half[i])}")
            if i == 0 or i == (len(frq) - 1):
                y_new += (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / Fs) * x + a[i])
            else:
                y_new += 2 * (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / Fs) * x + a[
                    i])  # I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
        i += 1



    plt.plot(x, y_new)  #################################################

    # creating the difference func between original and reconstructed signal
    i = 0
    diff = []
    while (i < len(y_new)):
        diff.append(y[i] - y_new[i])
        i += 1

    print(f"VAR of diff is: {stat.variance(diff)} and Estimation is: {stat.mean(diff)} ,  {stat.fmean(diff)}")

    plt.plot(x, diff)  ###################################################
    plt.legend(["first one", "second one", "difference"])
    plt.show()

    # plotting the spectrum

    plt.plot(frq, abs(Y_half))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.title("Amplitude of fft")
    plt.show()

    # ploting the phase of the fft

    plt.plot(frq, a_half)
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Arg (frq)')
    plt.title("Phase of fft")
    plt.show()


def main():
    SLTM_ML("GOOG")

if __name__ == "__main__":
    main()