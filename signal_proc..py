def DFT(y):
    """
      Function to calculate the
      discrete Fourier Transform
      of a 1D real-valued signal x
      """

    N = len(y)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    yf = np.dot(e, y)
    return yf


def IDFT(yf):
    """
      Function to calculate the
      discrete Fourier Transform
      of a 1D real-valued signal x
      """

    N = len(yf)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)

    y = np.dot(e, yf) / N

    return y


def PSD(func):
    new_func = func * np.conj(func)
    return new_func



# TODO - instead of this function need to make a tylor function to build the polinom
def zeros_builder_based_on_samples(x, func):
    i = 0
    zeros = 1
    while (i < len(func)):
        zeros *= (x - i)
        i += 1
    return zeros



def FWHM(frq, absY_half):
    HM = (max(absY_half) - absY_half[len(absY_half) - 1]) / 2
    index_of_maximal_spectrum_val = absY_half.tolist().index(max(absY_half))
    print(f"The index of maximal value is: {index_of_maximal_spectrum_val}")
    i = 0
    entry1 = 0
    entry2 = 0
    while i < len(absY_half):

        if absY_half[i] >= HM and i <= index_of_maximal_spectrum_val and not (entry1):
            print(
                f"\nFRQ1 -> the position of y2, x2 in calc of the linear func will be {i}, {absY_half[i]},{HM}  and y1, x1 will be {i - 1} accordingly \n")
            m = (absY_half[i] - absY_half[i - 1]) / (frq[i] - frq[i - 1])
            b = absY_half[i] - m * frq[i]
            frq_1 = (HM - b) / m

            entry1 = 1

        if absY_half[i] <= HM and i >= index_of_maximal_spectrum_val:
            print(
                f"\nFRQ2 -> the position of y2, x2 in calc of the linear func will be {i} and y1, x1 will be {i - 1} accordingly \n")
            m = (absY_half[i] - absY_half[i - 1]) / (frq[i] - frq[i - 1])
            b = absY_half[i] - m * frq[i]
            # frq_2 = (HM - b) / m
            frq_2 = frq[i - 1]  #### THIS IS MOST CERTAINLY NOT CORRECT - WE ASSUME THAT THE HM IS WHERE THE MAXIMUM IS
            break

        i += 1
    delta_f = frq_2 - frq_1
    return delta_f
