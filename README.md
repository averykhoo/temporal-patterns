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
*   support removal
*   try auto-correlation?
    *   first normalize months
*   the "fraction" stuff is basically a slow fourier transform to a specific set of frequencies
    *   non-linear because of months (and years)
    *   can it be generalized?
    *   any other interesting periods to consider
*   need to detect patterns that repeat inconsistently over time
    *   how about change points?
    *   sliding window to detect subsequence matches?
*   need to aggregate before doing stats?
    *   otherwise a few things in an hour doesn't meet the criteria for once a month
    *   what's the threshold for this?
    *   aggregate anything less than 1% of the fraction?
*   don't consider patterns unless the overall duration is at least 4x pattern length
    *   add warning if plotting?
*   don't consider patterns with a period of less than an hour
    *   add warning if plotting?
*   do stats for session durations and event density
*   visualizations better than numbers
    *   24h polar plot for days
    *   table heatmap plot
        *   month/hour heatmap for years
        *   month/week heatmap for years
*   use ann library for efficient vector lookups 
    *   try [scann](https://github.com/google-research/google-research/tree/master/scann)
    *   or [annoy](https://anaconda.org/conda-forge/python-annoy)
*   anomalies
    *   need to remove the event before seeing how anomalous it is
    *   how to handle zero probability?
    *   how to collate over multiple patterns?
*   likelihoods
    *   scale by number of matches? n-th root?
    *   sorting?
    