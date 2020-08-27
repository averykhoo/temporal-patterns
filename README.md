#   temporal_patterns

##  todo
*   how to find approximate patterns?
*   the "fraction" stuff is basically a slow fourier transform
    *   non-linear because of months
    *   can it be generalized?
    *   any other interesting periods to consider
*   need to detect patterns that repeat consistently over time
    *   how about change points?
    *   sliding window?
*   need to aggregate before doing stats?
    *   otherwise a few things in an hour doesn't meet the criteria for once a month
    *   what's the threshold for this?
    *   aggregate anything less than 1% of the fraction?
*   remove anomalies?
    *   huge spike in data
    *   missing data
    *   noise
