
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

