
# ########################################

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
#
# y_mean = sum(y)/len(y)
# print (f"The mean of all values of the given function PLUS zeros at the end is: {y_mean}")
# y = y - y_mean
#
# # y = np.sinc(np.exp(1j*t))
#
# #y=0
# #i=0
# # while (i<1):
# #       f = random.random()/2                  #creating the function
# #       a = random.randint(1,10)
# #       dc = random.randint(1,10)
# #       y += dc + a * np.sin(2*np.pi*f*t)
# #       i+=1
#
# plt.plot(t,y)
#
# #print(f, a, dc)
# # plt.show()
#
# #########################          ADDING WHITE NOISE
# # mean = 0
# # std = 0
# # num_samples = int(Fs/Ts)
# # # y += np.random.normal(mean, std, size=num_samples)
# # plt.plot(t,y)
# plt.show()
#
# ##########
# multiplications_number_concat = 1
# y_concat = y.copy()
# y = np.hstack([y_concat] * multiplications_number_concat)
#
# multiplications_number_zeros = 1                                                                # for better characterization in fft
#
# number_of_zeros = (multiplications_number_zeros-1)*len(y)
#
# i=0
# while i < number_of_zeros:                                                                #inserting zeros at the end of y_high function vector to maitain a better freq picture.
#       y = np.append(y, 0)
#       i= i+1
#
# mult_num = multiplications_number_zeros*multiplications_number_concat
#
# ##########
# n = len(y)                                      # length of the signal we created = Fs/Ts
#
# k = np.arange(n)                                #basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
# T = n*Ts                                        # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
# frq = k/T                                       # two sides frequency range
# frq = frq[:len(frq)//2]                         # one side frequency range
# Y = np.fft.fft(y)                               # dft and normalization
# Y_half = Y[:n // 2]
# a = np.angle(Y)
#
# a_half = a[:n // 2]
#
#
#
# y_new = 0
# The_length_I_want_to_see = Fs*2 ################ DO THE SAME IN THE TESTING SIGNAL TO CHECK IT OUT!!!!!!!
# steps = The_length_I_want_to_see/len(y)
# x = np.arange(0,The_length_I_want_to_see,steps) # BUILDING THE EXACT SAME VECTOR THAT WAS FOR THE ORIGINAL SIGNAL - t....
# i = 0
# max_val_abs_Y = max(abs(Y_half))
# while (i< len(frq)):
#       if (abs(Y_half[i]) > 0.001*max_val_abs_Y):            # whoever answers to the conditions will be represented in the recreated function
#
#             print(f"\nwe have frequency {i/(Fs*mult_num)} with the value of {abs(Y_half[i])} which is {abs(Y_half[i])/sum(abs(Y_half))*100}% from the total")
#
#             if i == 0 or i == (len(frq)-1):
#                   y_new += (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / (Fs*mult_num)) * x + a[i])
#             else:
#                   y_new += 2 * (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / (Fs*mult_num)) * x + a[i])    #I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
#
#       i+=1
#
#
# plt.plot(x,y_new)
# plt.legend(["first one", "second one"])
# plt.show()
#
# rms = np.sqrt(np.mean(abs(Y_half)**2))
# print(f"the RMS value of all values is: {rms}")
# print(abs(Y_half))
# plt.stem(frq, abs(Y_half))                             # plotting the spectrum
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
# plt.title("Amplitude of fft")
# plt.show()
#
#
#
# plt.plot(frq, a_half)                                  #ploting the phase of the fft
# plt.xlabel('Freq (Hz)')
# plt.ylabel('Arg (frq)')
# plt.title("Phase of fft")
# plt.show()


###     REAL   CODE   BELOW:


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
