import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.neighbors import KernelDensity

N = 1000
np.random.seed(1)
X = np.concatenate((np.random.normal(15, 10, int(0.3 * N)),
                    np.random.normal(75, 20, int(0.7 * N))))[:, np.newaxis]
X = np.array([x[0] for x in X if 0 <= x <= 100]).reshape(-1, 1)


def draw_histograms(data, labels, num_bins=40, font_size=11):
    assert len(labels) == len(data)
    min_val = 0.0
    max_val = 100.0

    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(nrows=len(data), ncols=1)
    for (x_mean, label, x), axis in zip(sorted(zip(map(np.mean, data), labels, data)),
                                        axes.flatten() if len(data) > 1 else [axes]):
        # plot histogram
        n, bins, patches = axis.hist(x, num_bins, density=0, range=(min_val, max_val), color='blue', alpha=0.2,
                                     label=label)

        plot_range = np.linspace(min_val, max_val, 1000)[:, np.newaxis]
        for k in ['gaussian', 'tophat', 'linear', 'epanechnikov', 'exponential', 'cosine']:
            kde = KernelDensity(kernel=k, bandwidth=3).fit(X)
            log_dens = kde.score_samples(plot_range)
            plt_vals = np.exp(log_dens) * (len(x) * (float(max_val - min_val) / num_bins))
            plt_vals[0] = 0
            plt_vals[-1] = 0
            axis.plot(plot_range[:, 0], plt_vals, label=k)

        # plot mean
        axis.axvline(x=x_mean, color='black', linestyle='--')

        # transform data to logit space
        l_data = scipy.special.logit(X / 100)
        l_mean = np.mean(l_data)
        l_sigma = np.sqrt(np.var(l_data))
        l_range = (l_mean - 4 * l_sigma, l_mean + 4 * l_sigma)

        # plot normal distribution in logit space
        l_curve_x = np.linspace(l_range[0], l_range[1], 1000)  # 1000 plot points
        l_curve_y = len(l_data) * ((l_range[1] - l_range[0]) / num_bins) * scipy.stats.norm.pdf(l_curve_x, l_mean,
                                                                                                l_sigma)

        # transform back into real space
        axis.plot(scipy.special.expit(l_curve_x) * 100, l_curve_y, color='black', linestyle='-', linewidth=1,
                  label='logit-norm')

        # skew-normal
        x2_skew = scipy.stats.skew(x)
        x2_skew, x2_mean, x2_sigma = scipy.stats.skewnorm.fit(x, x2_skew)
        curve_x = np.linspace(min_val, max_val, 1000)
        curve_y = scipy.stats.skewnorm.pdf((curve_x - x2_mean) / x2_sigma, x2_skew) / x2_sigma
        axis.plot(curve_x, len(x) * (max_val / num_bins) * curve_y, color='black', linestyle='--', linewidth=1,
                  label='skewnorm')
        # axis.axvline(x=x2_mean, color='g', linestyle='-', lw=4)

        # gaussian
        curve_x = np.linspace(min_val, max_val, 1001)
        n_plot = len(x) * (max_val / num_bins) * scipy.stats.norm.pdf(curve_x, x_mean, np.sqrt(np.var(x)))
        axis.plot(curve_x, n_plot, color='black', linestyle='-.', linewidth=1, label='gaussian')

        # shade cutoff area
        cut = 30  # score at which to cut off
        axis.axvline(x=cut, color='0.7', linestyle='-')
        axis.axvspan(0, cut, color='0.4', alpha=0.5)
        count_remaining = sum(val > cut for val in x)
        label += ': %d/%d (%0.2f%%) remaining' % (count_remaining, len(x), count_remaining * 100.0 / len(x))

        # labels and formatting
        axis.set_title(label, fontsize=font_size + 2)
        axis.set_ylabel('count', fontsize=font_size)
        axis.set_xlabel('score', fontsize=font_size)
        axis.grid(True)
        for tick_label in axis.get_yticklabels():
            tick_label.set_fontsize(font_size - 1)
        for tick_label in axis.get_xticklabels():
            tick_label.set_fontsize(font_size - 1)

        # fix y tick label to be actual amounts not probabilities
        tick_labels = [int(tick) if i % 4 == 0 else '' for i, tick in enumerate(axis.get_yticks())]
        axis.set_yticklabels(tick_labels)

    fig.tight_layout()
    plt.legend()

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()


draw_histograms([X], ['x'])
