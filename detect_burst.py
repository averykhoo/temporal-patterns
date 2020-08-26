import numpy as np
import pandas as pd

from .burst_detection import burst_detection, enumerate_bursts, TIME_BINS


def gen_burst_df(timestamps, srcip, ip_of_interest, granularity):
    """
    :param timestamps: Series of timestamps with datetime64[ns]
    :param srcip: Series of IP addresses
    :param ip_of_interest: IP to subset df on
    :param granularity : List of possible values : ['days', 'hours', 'minutes', 'seconds']
    :return: Reformed dataframe to process the values on
    """
    assert granularity in TIME_BINS

    df = pd.DataFrame({'timestamp': timestamps, 'srcip': srcip})
    df['time_bin'] = df['timestamp'].apply(lambda x: x.strftime(TIME_BINS[granularity]))

    df_left = df.groupby(['time_bin'])['srcip'].count()
    df_left = df_left.reset_index()
    df_subset = df[df.srcip == ip_of_interest]
    df_right = df_subset.groupby(['time_bin'])['srcip'].count().reset_index()

    final_df = pd.merge(df_left, df_right, how='left', left_on=['time_bin'], right_on=['time_bin'])
    final_df.columns = ['time_bin', 'total_count', 'target_count']
    final_df = final_df.fillna(0)
    return final_df


def detect_bursts(timestamps, srcip, ip_of_interest='', bursts_col=None, s=1.1, g=2, smoothing=5, granularity='hour'):
    """
    Returns a list of values which can be concatenated by axis = 1 with the original dataframe to
    determine if its a burst or not
    :param timestamps: Series of timestamps with datetime64[ns]
    :param srcip: Series of IP addresses
    :param ip_of_interest: IP to subset df on
    :param bursts_col: Series column which determines if the row is part of a burst or not
    :param s: resolution of state jumps; higher s --> bigger jumps between states
    :param g: difficulty of moving up a state; larger gamma --> harder to move up states
    :param smoothing: Window of number of points used for smoothing 
    :param granularity: List of possible values : ['day', 'hour', 'minutes', 'seconds']
    :return bursts_col: If none:
                        Series of nan, 0 , 1 corresponding to the timestamps where
                        there was a burst for the particular IP.
                        nan refers to IPs that were not in considerations
                        If not none:
                        Same as None but the elements which have been filled
    """
    if s == 1:
        raise ValueError('S cannot take the value of 1')

    if ip_of_interest not in list(srcip):
        raise ValueError('ip_of_interest: {0} not present in dataframe'.format(ip_of_interest))

    if granularity not in TIME_BINS:
        raise ValueError('granularity not in list of possible values {0}'.format(repr(TIME_BINS)))
    # Generate the dataframe to process on
    df = gen_burst_df(timestamps, srcip, ip_of_interest, granularity)
    targets = df.target_count.astype(float)
    total = df.total_count.astype(float)

    n = len(total)  # number of timepoints

    # find the optimal state sequence (using the Viterbi algorithm)
    [states, _, _, prob] = burst_detection(targets, total, n, s, g, smooth_win=smoothing)

    # create label
    label = 's={0}, g={1}'.format(s, g)

    # enumerate the bursts
    bursts = enumerate_bursts(states, label)
    ranges = zip(bursts['begin'], bursts['end'])

    # Portion to generate the burst column if it does not exist
    if not bursts_col:
        bursts_col = pd.Series([np.nan] * len(srcip), name='bursts')

    srcip_idx = srcip[srcip == ip_of_interest].index
    for idx in srcip_idx:
        bursts_col.loc[idx] = 0

    # Populate the burst list
    timestamps_subset = timestamps[srcip_idx]
    bursts_col = _populate_bursts(df, ranges, bursts_col, timestamps_subset, granularity)
    return bursts_col


def _populate_bursts(bursts_df, ranges, bursts_col, timestamps, granularity):
    """
    Sets the indexes at which there is a burst based on the original dataframe
    :param bursts_df: DataFrame returned by _gen_burst_df
    :param ranges: list of tuples of start and end position of bursts
    :param bursts_col: Series of values with the relevant src ip values
                        being 0 and the rest being NAN if uninitialized
    :param timestamps: timestamp series from unprocessed dataframe in detect_bursts
    :param granularity: List of possible values : ['day', 'hour', 'minutes', 'seconds']
    :return: Updated values of burst_cols with the relevant values being 1
    """
    # Set burst situations to 1
    # Empty df to store all the rows that would be 1
    df_burst_bins = pd.DataFrame(columns=['time_bin'])
    idx_timestamp_tup = zip(timestamps.index, timestamps)
    for start, end in ranges:
        burst_date_hr = bursts_df.iloc[start:end][['time_bin']]
        df_burst_bins = pd.concat((df_burst_bins, burst_date_hr))

    time_bins = df_burst_bins['time_bin'].unique()
    for idx, timestamp in idx_timestamp_tup:

        if granularity == 'hour':
            if '{0} {1:02}'.format(str(timestamp.date()), timestamp.hour) in time_bins:
                bursts_col.loc[idx] = 1

        elif granularity == 'minutes':
            if '{0} {1:02}:{2:02}'.format(str(timestamp.date()), timestamp.hour, timestamp.minute) in time_bins:
                bursts_col.loc[idx] = 1

        elif granularity == 'seconds':
            if timestamp in time_bins:
                bursts_col.loc[idx] = 1
    return bursts_col
