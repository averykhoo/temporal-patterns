#   temporal_patterns

*   given some datetimes 
    *   visualize the temporal patterns
        *   daily, weekly, monthly, yearly
    *   produce some vectors that can be used to find similar patterns via nearest neighbor
        *   similar patterns
        *   similar patterns with a phase difference
        *   similar overall
    *   find anomalies
        *   low likelihood events
        *   should find groups of events instead of single events?
        

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
    *   full week???
    *   24h polar plot for days
    *   day/hour heatmap for weeks, month/week heatmap for years
        *   https://infovis-mannheim.de/viavelox/
        *   https://infovis-mannheim.de/viavelox/assets/img/matrix.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/patterns1.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/patterns2.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/design1.jpg
        *   https://infovis-mannheim.de/viavelox/assets/img/design2.jpg
        *   https://www.vizwiz.com/2018/06/tfl-cycle-hire.html
*   convert pattern (eg fraction) into vector
    *   dimension=360 should be enough, resolution is:
        *   years:  1 day
        *   months: 2 hours
        *   weeks: 28 minutes
        *   days:   4 minutes
        *   or maybe have variable dimension for each?
    *   can rotate to match patterns that are offset
        *   either store many more patterns or do many more lookups
    *   can use ann library 
        *   try [scann](https://github.com/google-research/google-research/tree/master/scann)
        *   or [annoy](https://anaconda.org/conda-forge/python-annoy)
    *   lookup similarity separately for week, month, year, etc
*   anomalies
    *   need to remove the event before seeing how anomalous it is
    *   how to handle zero probability?
    *   how to collate over multiple patterns?
    