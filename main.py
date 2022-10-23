from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import datetime as dt


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from yfinance import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random


from datetime import date
import yfinance as yf
import scipy.fft
import numpy as np
import matplotlib.pyplot as plotter
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from datetime import date
from scipy import signal
import matplotlib.pyplot as plt
import math
import cmath
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

def main():
    pass

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

def ZeroToMastery(huge_vector):
    #set random seed
    tf.random.set_seed(42)

    #1. create the model using sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    #2. compile the model
    model.compile(loss = tf.keras.losses.mae,
                  optimizer = tf.keras.oprtimizers.SGD(),
                  metrics = ["mae"])

    #3. fit the model
    model.fit(x, y, epochs= 100)

def SLTM_ML(company_tag: str):

    # Load data
    company = yf.Ticker(company_tag)

    start = dt.datetime(2021, 9, 1)
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





###############################################################

# BELOW IS THE ATTEMPT OF CREATING STOCK BY NITZAN'S THEORY:

###############################################################

# Fs = 20  # SAMPLE FREQUENCY!
# Ts = 1/50 # what will be the sampling rate when considering THE FUCKING FREQUENCIES OF THE SYSTEM.
# #more than twice the biggest freq - since the spectrom is devided into two and needs to include the biggest freq.
# t = np.arange(0,5,Ts) # time vector
#
# print(Fs/Ts)
# ff1 = 0.0231   # frequency of the signal and not OMEGA
# a1 = 1
# ff2 = 0.0523
# a2 = 1
# ff3 = 0.0056
# a3 = 1
# ff4 = 0.0079
# a4 = 1
# ff5 = 0.0133
# a5 = 1
# dc = 17.3293
#
# y = dc + a1*np.sin(2*np.pi*ff1*t) + a2*np.sin(2*np.pi*ff2*t) #+ a3*np.sin(2*np.pi*ff3*t) + a4*np.sin(2*np.pi*ff4*t) + a5*np.sin(2*np.pi*ff5*t)

####################################################################

#BELOW IS THE ATTEMPT OF MANIPULATING THE FRQ OF THE STOCK:

####################################################################
# new_value_quantity = len(y_high_final)
#
# """using FFT here and not calculating by hand:"""
#
# yf = np.fft.fft(y_high_final)  # Here we get good values for the
# xf = np.fft.fftfreq(len(yf))
#
# plt.plot(xf, yf)  #####AMAZING... SINC FUNC
# plt.show()
#
# """plotting something"""
# sampling_rate = 1.0
# # calculate the frequency
# N = len(yf)  # when fft is done we need to take the whole new signal range not the original
# print("len of yf is:::::::::::::::", N)
# n = np.arange(N)
# T_sampling = 1.0
# freq = n / T_sampling
#
# ############################################
#
# """correcting the FOURIER TRANSFORM from symmetrical to a good one with proper values"""
# n_oneside = N // 2
# # get the one side frequency
# f_oneside = freq[:len(freq) // 2]
#
# # normalize the amplitude
# yf_func_oneside = yf[:n_oneside] / n_oneside
#
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.stem(f_oneside, abs(yf_func_oneside), 'b', markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('DFT Amplitude |X(freq)|')
#
# plt.show()
#
#
# plotting_regular_functions_abs_val(f_oneside, abs(yf_func_oneside), 120)
# no_hill_func = yf_func_oneside.copy()
# number_of_hills = 0
# number_of_hills_wanted = 4
# number_of_freq_of_zero = 0
# i = 1
# while (i < len(no_hill_func)):
#     if (number_of_hills == number_of_hills_wanted):
#         no_hill_func[i] = 0
#         number_of_freq_of_zero += 1
#     elif (i == len(no_hill_func) - 1):
#         no_hill_func[i] = 0
#         break
#     elif (abs(no_hill_func[i]) < abs(no_hill_func[i + 1]) and abs(no_hill_func[i]) < abs(no_hill_func[i - 1])):
#         number_of_hills += 1
#
#     i += 1
#     print(f"the number of freqs included not zeros:{number_of_values_wanted - number_of_freq_of_zero}")
# no_hill_func_fft = np.fft.ifft(
#     no_hill_func)  ######THIS DOESNT WORK BECAUSE I NEED ONLY THE NUMBER OF SAMPLES AND I HAVE GOT A LOT MORE!
# x_no_hill = np.arange(len(no_hill_func_fft))
#
# # plotting_regular_functions_abs_val(x_no_hill, no_hill_func_fft, 170)
# # ploting_stem_functions_abs_val(x_no_hill, no_hill_func_fft, 170)
#
#
# """SAMPLING THE FREQUENCY REPRESENTATION OF THE SIGNAL"""
#
# # number of samples in freq
# sn = 2
# # sampling interval
# fs = N // sn
# f = np.arange(0, N, fs)
# newly_yf_TEST2 = []
# fs_tracker = 1
# counter_of_samles_so_far = 0
# i = 0
# while i < len(yf):
#     if i == fs_tracker:
#         newly_yf_TEST2.append(yf[i])
#         counter_of_samles_so_far += 1
#         fs_tracker += fs
#     i += 1
# print(f"counter of samples so far:::::::{counter_of_samles_so_far}")
# # print_whole_list(newly_yf_TEST2)
#
# newly_y_TEST2 = ifft(newly_yf_TEST2)
# newly_x_TEST2 = np.arange(len(newly_y_TEST2))
#
# # plotting_regular_functions_abs_val(newly_x_TEST2, newly_y_TEST2, 70)
#
#
# """HERE I CLEAR THE FREQUENCY FUNCTION FROM NOISE SUPPOSEDLY"""
# """recreating the sinusoidal function to recommend on buying or selling"""
# i = 0  # When concating - clears all but spikey freqs
# sum = 0
# number_of_frequencies_wanted = 1000
# ratio_between_neighbors = 1
# yf_clean = yf_func_oneside.copy()
# is_it_first_freq = 0
# while i < yf_func_oneside.size:
#
#     if (sum == number_of_frequencies_wanted):
#         yf_clean[i] = 0
#         i += 1
#         continue
#     if (i == 0 or i == yf_func_oneside.size - 1):
#         yf_clean[i] = 0
#         i += 1
#         continue
#     elif ((abs(yf_func_oneside[i]) > ratio_between_neighbors * abs(yf_func_oneside[i - 1]))
#           and (abs(yf_func_oneside[i]) > ratio_between_neighbors * abs(yf_func_oneside[i + 1]))):
#         sum += 1
#         if (is_it_first_freq == 0):
#             first_freq_size = abs(yf_func_oneside[i])
#         is_it_first_freq = 1
#         print(
#             f"THE NUMBER IS {i} AND IT POWER IS {abs(yf_func_oneside[i])} AND THE RATIO TO FIRST FREQ IS: {first_freq_size / abs(yf_func_oneside[i])}")
#         i += 1
#     else:
#         yf_clean[i] = 0
#         i += 1
#
# print(f"Number of spikes found, INCLUDING those who are neighbors : {sum}")
#
# yf_clean_extra_zeros = yf_clean.copy()
# # i = 0
# # while i < yf_clean.size:
# #       if yf_clean[yf_clean.size - 1 - i] == 0:
# #             yf_clean_extra_zeros = np.delete(yf_clean_extra_zeros, yf_clean.size - 1 - i)
# #       else:
# #             break
# #       i += 1
#
# # print(np.abs(yf_clean_extra_zeros))
#
#
# # i = 0
# # final_func_TEST = []
# # while (i<len(y_recreation)-1):                                                #HERE I TRY TO PAD THE FREQ WITH ZEROS INBETWEEN
# #       if i%2 == 0:
# #             final_func_TEST.append(yf[i])
# #       else:
# #             final_func_TEST.append(0)
# #       i+=1
#
#
# # i = 0
# # l = 0
# # old_freq = 0
# # final_function_output = 0
# # length_of_output_wanted = 1*len(yf)
# # x_of_final = np.arange(0, length_of_output_wanted,1)              #the plotted range - WE WANT THE FUTURE!       # 1/T_sampling gives us way better percission
# # print(len(yf_func_oneside))
# # while i < len(yf_func_oneside):
# #       if (abs(yf_func_oneside[i]) > 0):
# #             final_function_output += 2*(abs(yf_func_oneside[i])) * np.exp(2j*np.pi*i* x_of_final/length_of_output_wanted)
# #
# #             print(f"the {l}th is {i}")
# #             l += 1
# #       i+=1
# #
# #
# #
# #
# # plt.plot(x_of_final, final_function_output)
# # plt.show()
# #############################################################################3
#
# n = len(yf)  # length of the signal we created
# k = np.arange(n)  # basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
# T = n / n  # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
# frq = k / T  # two sides frequency range
# frq = frq[:len(frq) // 2]  # one side frequency range
# Y_half = yf / n  # dft and normalization
# Y_half = Y_half[:n // 2]
#
# plt.plot(frq, abs(Y_half))  # plotting the spectrum
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
# plt.title('THE FUCKING GREATEST FUCKING GRAPH EVER!!!!!')
# plt.show()
#
# y = 0
# length_wanted = n * 3
# x = np.arange(0, length_wanted, 1)
# i = 0
# while (i < len(frq)):
#     if (abs(Y_half[i]) > 0):  # whoever answers to the conditions will be represented in the recreated function
#         print(f"we have {i} with power of {abs(Y_half[i])}")
#         y += 2 * (abs(Y_half[i])) * np.exp(2j * np.pi * i * x / (n / 2))  # I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
#     i += 1
#
# plt.plot(x, y)
# plt.show()
