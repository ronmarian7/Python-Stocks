def plotting_regular_functions_abs_val(x, y, size):
    plt.figure(num=-10000.0, dpi=size)
    plt.plot(x, np.abs(y), 'k')
    plt.show()


def plotting_stem_functions_abs_val(x, y, size):
    plt.figure(num=-100.0, dpi=size)
    plt.stem(x, np.abs(y), use_line_collection=True)
    plt.show()


def print_whole_list(list):
    i = 0
    while i < len(list):
        print(list[i])
        i += 1



def poles_builder_based_on_samples(x, func, value_index):
    i = 0
    poles = 1
    while (i < len(func)):
        if (i != value_index):
            poles *= (value_index - i)
        i += 1
    return poles
