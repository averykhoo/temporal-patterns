# This module creates definitions for implementing Kleinberg's burst detection analysis on batched data.

import numpy as np
import pandas as pd
from scipy.special import comb

TIME_BINS = {'days':    '%Y-%m-%d',
             'hours':   '%Y-%m-%d %H',
             'minutes': '%Y-%m-%d %H:%M',
             'seconds': '%Y-%m-%d %H:%M:%S',
             }


# define the transition cost tau: cost of switching states
# there's a cost to move up states, no cost to move down
# based on definition on pg. 8
# inputs
#   i1: current state
#   i2: next state
#   gam: gamma, penalty for moving up a state
#   n: number of timepoints
def tau(i1, i2, gamma, n):
    if i1 >= i2:
        return 0
    else:
        return (i2 - i1) * gamma * np.log(n)


# define the fit cost: goodness of fit to the expected outputs of each state
# based on equation on bottom of pg. 14
#    d: number of events in each time period (1xn)
#    r: number of target events in each time period (1xn)
#    p: expected proportions of each state (1xk)
def fit(d, r, p):
    a = -np.log(comb(d, r))
    b = -np.log(p) * r
    c_val = -np.log(1 - p) * (d - r)
    goodness_val = a + b + c_val
    return goodness_val


# define the burst detection function for a two-state automaton
# inputs:
#   r: number of target events in each time period (1xn)
#   d: number of events in each time period (1xn)
#   n: number of timepoints
#   s: multiplicative distance between states
#   gamma: difficulty to move up a state
#   smooth_win: width of smoothing window (use odd numbers)
# output:
#   q: optimal state sequence (1xn)
def burst_detection(r, d, n, s, gamma, smooth_win):
    assert smooth_win % 2 == 1
    assert s > 0
    assert gamma > 0

    k = 2  # two states

    # smooth the data if the smoothing window is greater than 1
    if smooth_win > 1:
        temp_p = r / d  # calculate the proportions over time and smooth
        temp_p = temp_p.rolling(window=smooth_win, center=True).mean()
        # update r to reflect the smoothed proportions
        r = temp_p * d
        real_n = sum(~np.isnan(r))  # update the number of timepoints
    else:
        real_n = n

    # calculate the expected proportions for states 0 and 1
    p = dict()
    p[0] = np.nansum(r) / float(np.nansum(d))  # overall proportion of events, baseline state
    p[1] = max(0.99999, p[0] * s)  # proportion of events during active state (can't be bigger than 1)

    # Initialize matrices to hold the costs and optimal state sequence
    cost = np.full([n, k], np.nan)
    q_t = np.full([n, 1], np.nan)

    # Use the Viterbi algorithm to find the optimal state sequence
    for t in range(int((smooth_win - 1) / 2), (int((smooth_win - 1) / 2)) + real_n):

        # Calculate the cost to transition to each state
        for j in range(k):
            # For the first timepoint, calculate the fit cost only
            if t == int((smooth_win - 1) / 2):
                cost[t, j] = fit(d[t], r[t], p[j])

            # For all other timepoints, calculate the fit and transition cost
            else:
                cost[t, j] = tau(q_t[t - 1], j, gamma, real_n) + fit(d[t], r[t], p[j])

        # Ddd the state with the minimum cost to the optimal state sequence
        # If infinity assume is state 0 
        if np.isinf(min(cost[t, :])) or np.isnan(min(cost[t, :])):
            q_t[t] = 0
        else:
            # av edit: I assume you want the first state?
            # there can be multiple states so i'm taking the min of whatever's returned
            q_t[t] = np.min(np.where(cost[t, :] == min(cost[t, :])))
    return q_t, d, r, p


# define a function to enumerate the bursts
# input:
#   q: optimal state sequence
# output:
#   bursts: dataframe with beginning and end of each burst
def enumerate_bursts(q, label):
    bursts = pd.DataFrame(columns=['label', 'begin', 'end', 'weight'])

    if len(q) < 2:
        return bursts

    b = 0
    burst = False
    for t in range(1, len(q)):

        if not burst and (q[t] > q[t - 1]):
            bursts.loc[b, 'begin'] = t
            burst = True

        elif burst and (q[t] < q[t - 1]):
            bursts.loc[b, 'end'] = t
            burst = False
            b = b + 1

    # if the burst is still going, set end to last timepoint
    if burst:
        bursts.loc[b, 'end'] = len(q) - 1

    bursts.loc[:, 'label'] = label

    return bursts


# define a function that finds the weights associated with each burst
# find the difference in the cost functions for p0 and p1 in each burst
# inputs:
#   bursts: dataframe containing the beginning and end of each burst
#   r: number of target events in each time period
#   d: number of events in each time period
#   p: expected proportion for each state
# output:
#   bursts: dataframe containing the weights of each burst, in order
def burst_weights(bursts, r, d, p):
    # loop through bursts
    for b in range(len(bursts)):

        cost_diff_sum = 0

        for t in range(bursts.loc[b, 'begin'], bursts.loc[b, 'end']):
            cost_diff_sum = cost_diff_sum + (fit(d[t], r[t], p[0]) - fit(d[t], r[t], p[1]))

        bursts.loc[b, 'weight'] = cost_diff_sum

    return bursts.sort_values(by='weight', ascending=False)

#
#
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import rcParams
#
# from .burst_detection import burst_detection, enumerate_bursts, TIME_BINS
# from .detect_burst import gen_burst_df
#
# sns.set_style("white")
# rcParams['font.size'] = 14
#
#
# # create a timeline of the bursts
# def _plot_burst_timeline(bursts, timepoints, label):
#     f, ax = plt.subplots(figsize=(8, 0.5))
#     ax.set(xlim=(0, timepoints), ylabel="", xlabel="")
#
#     # create boxes around bursting periods
#     for index, burst in bursts.iterrows():
#         # define outline positions
#         y = 0.25
#         xstart = burst['begin'] - 1
#         width = burst['end'] - burst['begin']
#
#         # draw rectangle
#         ax.add_patch(plt.Rectangle((xstart, y), width, height=0.5,
#                                    facecolor='#00bbcc', edgecolor='none', linewidth=1))
#
#     # remove borders
#     ax.xaxis.set_visible(False)
#     plt.yticks([0.5], [label], size=14)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#
#     # add a timeline
#     plt.axhline(0.5, linewidth=1, color='k', alpha=1, zorder=0.5)
#
#     plt.show()
#
#
# def plot_bursts(timestamps, srcip, ip_of_interest, s=1.1, g=2, smoothing=5, granularity='hour'):
#     df = gen_burst_df(timestamps, srcip, ip_of_interest, granularity)
#     targets = df.target_count.astype(float)
#     total = df.total_count.astype(float)
#     n = len(total)  # number of timepoints
#
#     # find the optimal state sequence (using the Viterbi algorithm)
#     [states, _, _, p] = burst_detection(targets, total, n, s, g, smooth_win=smoothing)
#
#     # create label
#     label = 's={0}, g={1}'.format(s, g)
#
#     # enumerate the bursts
#     bursts = enumerate_bursts(states, label)
#     ranges = zip(bursts['begin'], bursts['end'])
#
#     # Define the formats
#     print('Time format:{0} '.format(TIME_BINS[granularity]))
#     print('start\t\tend')
#     for start, end in ranges:
#         print('{0}\t{1}'.format(df.time_bin.iloc[start], df.time_bin.iloc[end]))
#     _plot_burst_timeline(bursts, n, label)
