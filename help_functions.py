
def building_polinom_for_given_func(x, func, zeros):
    i = 0
    polinom = 0
    while (i < len(func)):
        polinom += func[i] * (zeros / (x - i)) / (poles_builder_based_on_samples(x, func, i))
        i += 1
    return polinom
