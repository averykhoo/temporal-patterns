import math

try:
    from fastcache import clru_cache as lru_cache
except ModuleNotFoundError:
    from functools import lru_cache


def general_mean(*xs, dim=1):
    """
    https://en.wikipedia.org/wiki/Generalized_mean

    special cases:
    dim=-inf -> minimum
    dim=-1   -> harmonic mean
             -- geometric-harmonic mean would fit in here
    dim=0    -> geometric mean
             -- logarithmic mean would fit in here
             -- arithmetic-geometric mean would fit in here
    dim=1    -> arithmetic mean
    dim=2    -> root mean square (quadratic mean)
    dim=3    -> cubic mean
             -- contra-harmonic mean would fit in here
    dim=inf  -> maximum
    """
    if dim == 0:
        # geometric mean
        # return math.prod(xs) ** (1 / len(xs))  # `math.prod` is only available in Python 3.8 and above
        return math.exp(sum(map(math.log, xs))) ** (1 / len(xs))
    elif dim == math.inf:
        return max(xs)
    elif dim == -math.inf:
        return min(xs)
    return (sum(x ** dim for x in xs) / len(xs)) ** (1 / dim)


@lru_cache(maxsize=65536)
def log_mean(*xs):
    """
    generalized logarithmic mean
    https://en.wikipedia.org/wiki/Logarithmic_mean#Generalization

    in base 2, log_mean(1, 2) == 1, which breaks the math
    not sure if there's a similar problem in base e, but doesn't seem like it
    """
    xs = sorted(xs)  # helps with caching and dealing with duplicates
    if len(xs) == 0:
        raise TypeError('expected at least 1 arguments, got 0')
    elif xs[0] < 0:
        raise ValueError('cannot log a negative number')
    elif xs[0] == 0:
        # as xs[0] -> 0, log_mean -> 0, even though log(0) is undefined
        return 0
    elif xs[0] == xs[-1]:
        # by definition, min <= mean <= max, so if min == max, then min == mean
        # this also takes care of single-argument means
        return xs[0]
    elif len(xs) == 2:
        # https://en.wikipedia.org/wiki/Logarithmic_mean
        ret = (xs[1] - xs[0]) / (math.log(xs[1]) - math.log(xs[0]))
        assert xs[0] < ret < xs[-1], (xs, ret)
        return ret
    else:
        # https://www.survo.fi/papers/logmean.pdf (page 7, formula 14)
        ret = (len(xs) - 1) * (log_mean(*xs[1:]) - log_mean(*xs[:-1])) / (math.log(xs[-1]) - math.log(xs[0]))
        assert xs[0] < ret < xs[-1], (xs, ret)
        return ret


def contraharmonic_mean(*xs):
    numerator = 1
    denominator = 1
    for x in xs:
        numerator += x * x
        denominator += x
    return numerator / denominator


def geometric_harmonic_mean(*xs):
    """
    https://en.wikipedia.org/wiki/geometric-harmonic_mean
    """
    g_mean = general_mean(*xs, dim=0)
    h_mean = general_mean(*xs, dim=-1)

    while abs(g_mean - h_mean) / max(g_mean, h_mean) > 1e-15:
        g_next = math.sqrt(g_mean * h_mean)
        h_next = 2 / (1 / g_mean + 1 / h_mean)
        g_mean = g_next
        h_mean = h_next

    return g_mean


def arithmetic_geometric_mean(*xs):
    """
    https://en.wikipedia.org/wiki/arithmetic-geometric_mean
    """
    a_mean = general_mean(*xs)
    g_mean = general_mean(*xs, dim=0)

    while abs(g_mean - a_mean) / max(g_mean, a_mean) > 1e-15:
        a_next = (a_mean + g_mean) / 2
        g_next = math.sqrt(g_mean * a_mean)
        a_mean = a_next
        g_mean = g_next

    return g_mean
