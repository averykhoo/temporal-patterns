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
*   don't consider patterns unless the overall duration is at least 4x pattern length
*   don't consider patterns with a period of less than an hour
*   merge sessions (events within 48 hours)
    *   count sessions patterns too
        *   start, duration
*   visualizations better than numbers
    *   month/day/hour heatmap
        *   https://infovis-mannheim.de/viavelox/
        *   https://infovis-mannheim.de/viavelox/assets/img/matrix.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/patterns1.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/patterns2.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/design1.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/design2.jpg
        *   https://www.vizwiz.com/2018/06/tfl-cycle-hire.html
*   convert pattern (eg fraction) into vector
    *   maybe 480d for a month? (30 days *  24 hours)
    *   concatenate the 3 sets of the pattern, apply kde, then take the middle part
    *   can rotate to match patterns that are offset
        *   either store many more patterns or do many more lookups
    *   can use ann library 
        *   try [scann](https://github.com/google-research/google-research/tree/master/scann)
        *   or [annoy](https://anaconda.org/conda-forge/python-annoy)
*   lookup similarity separately for week, month, year, etc
    *   how to balance between the similarities?
    *   equal weight?
    *   harmonic/geometric mean?
    *   