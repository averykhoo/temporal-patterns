import math


def general_mean(*xs, dim=1):
    """

    :param xs:
    :param dim:
    :return:
    """
    if pow == 0:
        return math.exp(sum(map(math.log, xs))) ** (1 / len(xs))
    return (sum(x ** pow for x in xs) / len(xs)) ** (1 / dim)


def log_mean(*xs):
    # xs = sorted(xs)
    print(len(xs))
    if len(xs) == 0:
        raise TypeError('expected at least 1 arguments, got 0')
    elif xs[0] == xs[-1]:
        # by definition, min <= mean <= max, so if min == max, then min == mean
        return xs[0]
    elif 0 in xs:
        # technically it's undefined, but as any value tends towards zero, the logarithmic mean tends towards zero
        return 0
    elif len(xs) == 2:
        # https://en.wikipedia.org/wiki/Logarithmic_mean
        return (xs[0] - xs[1]) / (math.log(xs[0]) - math.log(xs[1]))
    else:
        # https://www.survo.fi/papers/logmean.pdf (page 7, formula 14)
        return (len(xs) - 1) * (log_mean(*xs[1:]) - log_mean(*xs[:-1])) / (math.log(xs[-1]) - math.log(xs[0]))
