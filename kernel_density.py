import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import scipy.stats
from sklearn.neighbors import KernelDensity


def plot_kde(xs, min_val, max_val, kernel='gaussian', bandwidth=None, n_samples=1001):
    assert max_val > min_val
    assert n_samples > 0

    # convert to numpy array if necessary
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
    if len(xs.shape) == 1:
        xs = xs.reshape(-1, 1)

    # default bandwidth will be 3% of the overall width
    if bandwidth is None:
        bandwidth = (max_val - min_val) * 0.03

    # fit kde kernel
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(xs)  # todo: what happens if this is empty

    # plot kde
    kde_xs = np.linspace(min_val, max_val, n_samples).reshape(-1, 1)  # x-values to sample
    log_density = kde.score_samples(kde_xs)  # log of y-values at each x
    kde_ys = np.exp(log_density)
    return kde_xs[:, 0], kde_ys


def plot_kde_modulo(xs, modulo, kernel='gaussian', bandwidth=None, n_samples=1000):
    assert n_samples > 0
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
        xs.reshape(-1, 1)

    xs_mod = xs % modulo  # mod the data
    xs_extended = np.concatenate((xs_mod - modulo, xs_mod, xs_mod + modulo))  # wrap around left and right
    kde_xs, kde_ys = plot_kde(xs_extended, 0, modulo, kernel=kernel, bandwidth=bandwidth, n_samples=n_samples + 1)
    kde_xs = kde_xs[:-1]  # don't include the modulo
    kde_ys = kde_ys[:-1] * 3  # scale up since this in the middle third
    return kde_xs, kde_ys


def draw_histograms(data, labels, min_val=0.0, max_val=100, num_bins=60):
    assert len(labels) == len(data)

    plt.cla()
    plt.clf()
    plt.close()

    _x_axis_label = 'score'
    _y_axis_label = 'count'
    _axis_font_size = 11
    _data_label_font_size = _axis_font_size + 2  # title in large font

    fig, axes = plt.subplots(nrows=len(data))  # , ncols=1)

    # sorts by x_mean
    for (x_mean, label, x), axis in zip(sorted(zip(map(np.mean, data), labels, data)),
                                        axes.flatten() if len(data) > 1 else [axes]):

        # filter to min/max
        x = np.array([elem[0] for elem in x if min_val <= elem <= max_val]).reshape(-1, 1)

        # plot histogram
        n, bins, patches = axis.hist(x, num_bins, density=0, range=(min_val, max_val), color='blue', alpha=0.2,
                                     label='histogram')
        assert len(bins) - 1 == len(n) == num_bins, (len(bins), len(n), num_bins)
        assert sum(n) == len(x)

        # plot kde
        for kernel in ['tophat', 'gaussian', 'linear', 'epanechnikov', 'exponential', 'cosine']:
            kde_xs, kde_ys = plot_kde(x, min_val=min_val, max_val=max_val, kernel=kernel)

            # scale the kde curve up to match the histogram
            kde_ys *= len(x) * float(max_val - min_val) / num_bins

            # plot and fill area
            axis.plot(kde_xs, kde_ys, linewidth=1, label=kernel)
            # axis.fill_between(kde_xs, kde_ys, alpha=0.2)

        # # transform data to logit space
        # # import scipy.special
        # l_data = scipy.special.logit((x - min_val) / (max_val - min_val))
        # l_mean = np.mean(l_data)
        # l_sigma = np.sqrt(np.var(l_data))
        # l_range = (l_mean - 4 * l_sigma, l_mean + 4 * l_sigma)
        #
        # # plot normal distribution in logit space
        # l_curve_x = np.linspace(l_range[0], l_range[1], 1000)  # 1000 plot points
        # l_curve_y = len(l_data) * ((l_range[1] - l_range[0]) / num_bins) * \
        #             scipy.stats.norm.pdf(l_curve_x, l_mean, l_sigma)
        #
        # # transform back into real space
        # # WARNING: the area under the transformed curve is NOT normalized
        # curve_x = (scipy.special.expit(l_curve_x) * (max_val - min_val)) + min_val
        # axis.plot(curve_x, l_curve_y, color='black', linestyle='-', linewidth=0.5, label='logit-norm')

        # gaussian
        curve_x = np.linspace(min_val, max_val, 1001)
        curve_y = len(x) * (max_val / num_bins) * scipy.stats.norm.pdf(curve_x, x_mean, np.sqrt(np.var(x)))
        axis.plot(curve_x, curve_y, color='black', linestyle='-.', linewidth=1, label='gaussian')

        # plot mean
        axis.axvline(x=x_mean, color='black', linestyle='-', label='mean')

        # skew-normal
        x2_skew = scipy.stats.skew(x)
        x2_skew, x2_mean, x2_sigma = scipy.stats.skewnorm.fit(x, x2_skew)
        curve_x = np.linspace(min_val, max_val, 1001)
        curve_y = scipy.stats.skewnorm.pdf((curve_x - x2_mean) / x2_sigma, x2_skew) / x2_sigma
        curve_y *= len(x) * (max_val / num_bins)
        axis.plot(curve_x, curve_y, color='black', linestyle='--', linewidth=1, label='skewnorm')

        # # plot skew-normal mean
        # axis.axvline(x=x2_mean, color='g', linestyle='-', lw=4)

        # # shade cutoff area
        # cut = 30  # score at which to cut off
        # axis.axvline(x=cut, color='0.7', linestyle='-')
        # axis.axvspan(min_val, cut, color='0.4', alpha=0.5)

        # labels and formatting
        axis.grid(True)
        axis.set_title(label, fontsize=_data_label_font_size)
        axis.set_ylabel(_y_axis_label, fontsize=_axis_font_size)
        axis.set_xlabel(_x_axis_label, fontsize=_axis_font_size)
        for tick_label in axis.get_yticklabels():
            tick_label.set_fontsize(_axis_font_size - 1)
        for tick_label in axis.get_xticklabels():
            tick_label.set_fontsize(_axis_font_size - 1)

        # # fix y tick label to be actual amounts not probabilities
        # tick_labels = [int(tick) if i % 4 == 0 else '' for i, tick in enumerate(axis.get_yticks())]
        # axis.set_yticklabels(tick_labels)
        axis.set_xlim(left=min_val, right=max_val)

    fig.tight_layout()
    plt.legend()

    plt.show()

    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    # generate random stuff
    N = 1000
    np.random.seed(1)
    X1 = np.concatenate((np.random.normal(15, 10, int(0.3 * N)),
                         np.random.normal(75, 20, int(0.7 * N))))

    X2 = np.concatenate((np.random.normal(25, 15, int(0.4 * N)),
                         np.random.normal(45, 10, int(0.6 * N))))

    # reshape
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)

    # plot the stuff
    draw_histograms([X1, X2], ['x1', 'x2'])
