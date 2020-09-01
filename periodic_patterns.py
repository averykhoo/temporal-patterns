import bisect
import datetime

import matplotlib.pyplot as plt

from kernel_density import plot_kde_modulo

days_of_week = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]

month_names = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
]

month_lengths = [
    31,
    28.25,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
]


class TimeStampSet:
    def __init__(self, default_session_length=datetime.timedelta(hours=48)):
        self.default_session_length = default_session_length

        self.timestamps = []

        self.hour_of_day = dict()

        self.day_of_week = dict()
        self.nth_day_of_week = dict()
        self.full_week = dict()

        self.day_fraction = dict()
        self.week_fraction = dict()
        self.two_week_fraction = dict()
        self.month_fraction = dict()
        self.two_month_fraction = dict()
        self.three_month_fraction = dict()
        self.six_month_fraction = dict()
        self.year_fraction = dict()
        self.two_year_fraction = dict()

        self.month = dict()
        self.month_part = dict()

        self.day_since_epoch = dict()
        self.week_since_epoch = dict()
        self.two_week_since_epoch = dict()
        self.month_since_epoch = dict()
        self.two_month_since_epoch = dict()
        self.three_month_since_epoch = dict()
        self.six_month_since_epoch = dict()
        self.year_since_epoch = dict()
        self.two_year_since_epoch = dict()

        self.__kdes = None

    def add(self, timestamp: datetime.datetime):
        # replace timezone
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

        self.__kdes = None

        # count
        timestamp_idx = len(self.timestamps)
        self.timestamps.append(timestamp)
        assert self.timestamps[timestamp_idx] is timestamp, 'race condition?'

        # time of day (hour)
        self.hour_of_day.setdefault(timestamp.hour, []).append(timestamp_idx)

        # day of week (starts at Monday)
        day_of_week = days_of_week[timestamp.weekday()]
        self.day_of_week.setdefault(day_of_week, []).append(timestamp_idx)

        # nth 7-day-period of month (starts at 1)
        n = ((timestamp.day + 1) // 7) + 1
        self.nth_day_of_week.setdefault((n, day_of_week), []).append(timestamp_idx)

        # n-th full week of month (starts at 1, can be 0)
        full_week = (timestamp.day + 6 - timestamp.weekday()) // 7
        self.full_week.setdefault(full_week, []).append(timestamp_idx)

        # day fraction (0 to less than 1)
        _start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        _end = (timestamp + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
        day_since_epoch = datetime.timedelta(seconds=timestamp.timestamp()).days  # 1970-01-02 -> 1
        self.day_fraction.setdefault(day_fraction, []).append(timestamp_idx)
        self.day_since_epoch.setdefault(day_since_epoch, []).append(timestamp_idx)

        # week fraction (0 to less than 1)
        week_fraction = (timestamp.weekday() + day_fraction) / 7
        week_since_epoch = (day_since_epoch + 3) // 7  # 1st Monday after epoch (1970-01-05) -> 1
        self.week_fraction.setdefault(week_fraction, []).append(timestamp_idx)
        self.week_since_epoch.setdefault(week_since_epoch, []).append(timestamp_idx)

        # fortnight fraction
        two_week_fraction = ((day_since_epoch + 3) % 14 + day_fraction) / 14
        two_week_since_epoch = week_since_epoch // 2
        self.two_week_fraction.setdefault(two_week_fraction, []).append(timestamp_idx)
        self.two_week_since_epoch.setdefault(two_week_since_epoch, []).append(timestamp_idx)

        # month fraction (0 to less than 1)
        _start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _end = (_start + datetime.timedelta(days=35)).replace(day=1)
        month_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
        month_since_epoch = (timestamp.year - 1970) * 12 + timestamp.month - 1
        self.month_fraction.setdefault(month_fraction, []).append(timestamp_idx)
        self.month_since_epoch.setdefault(month_since_epoch, []).append(timestamp_idx)

        # which month, and is it early or late in the month
        self.month.setdefault(month_names[timestamp.month - 1], []).append(timestamp_idx)
        if month_fraction < 0.33:
            self.month_part.setdefault('early', []).append(timestamp_idx)
        elif month_fraction < 0.67:
            self.month_part.setdefault('mid', []).append(timestamp_idx)
        else:
            assert month_fraction < 1
            self.month_part.setdefault('late', []).append(timestamp_idx)

        # bimonthly fraction (0 to less than 1)
        two_month_fraction = (month_since_epoch % 2 + month_fraction) / 2
        two_month_since_epoch = month_since_epoch // 2
        self.two_month_fraction.setdefault(two_month_fraction, []).append(timestamp_idx)
        self.two_month_since_epoch.setdefault(two_month_since_epoch, []).append(timestamp_idx)

        # quarterly fraction (0 to less than 1)
        three_month_fraction = (month_since_epoch % 4 + month_fraction) / 4
        three_month_since_epoch = month_since_epoch // 4
        self.three_month_fraction.setdefault(three_month_fraction, []).append(timestamp_idx)
        self.three_month_since_epoch.setdefault(three_month_since_epoch, []).append(timestamp_idx)

        # biannual fraction (0 to less than 1)
        six_month_fraction = (month_since_epoch % 6 + month_fraction) / 6
        six_month_since_epoch = month_since_epoch // 6
        self.six_month_fraction.setdefault(six_month_fraction, []).append(timestamp_idx)
        self.six_month_since_epoch.setdefault(six_month_since_epoch, []).append(timestamp_idx)

        # year fraction (0 to less than 1)
        _start = timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        _end = _start.replace(year=timestamp.year + 1)
        year_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
        year_since_epoch = timestamp.year - 1970
        self.year_fraction.setdefault(year_fraction, []).append(timestamp_idx)
        self.year_since_epoch.setdefault(year_since_epoch, []).append(timestamp_idx)

        # biennial fraction (0 to less than 1)
        two_year_fraction = (year_since_epoch % 2 + year_fraction) / 2
        two_year_since_epoch = year_since_epoch // 2
        self.two_year_fraction.setdefault(two_year_fraction, []).append(timestamp_idx)
        self.two_year_since_epoch.setdefault(two_year_since_epoch, []).append(timestamp_idx)

    def consecutive(self):
        periods_since_epoch = {
            'days':         self.day_since_epoch,
            'weeks':        self.week_since_epoch,
            'two_weeks':    self.two_week_since_epoch,
            'months':       self.month_since_epoch,
            'two_months':   self.two_month_since_epoch,
            'three_months': self.three_month_since_epoch,
            'six_months':   self.six_month_since_epoch,
            'years':        self.year_since_epoch,
            'two_years':    self.two_year_since_epoch,
        }

        out = dict()
        for period_name, period_dict in periods_since_epoch.items():
            buffer = []
            for _period in sorted(period_dict.keys()):
                if not buffer:
                    buffer.append(_period)
                elif _period - buffer[0] == 1:
                    buffer.append(_period)
                else:
                    if len(buffer) > 1:
                        out.setdefault(period_name, []).append(buffer)
                    buffer = []

        return out

    def fractions(self, dim=1000):
        if self.__kdes is not None:
            return self.__kdes

        period_fractions = {
            'days':         self.day_fraction,
            'weeks':        self.week_fraction,
            'two_weeks':    self.two_week_fraction,
            'months':       self.month_fraction,
            'two_months':   self.two_month_fraction,
            'three_months': self.three_month_fraction,
            'six_months':   self.six_month_fraction,
            'years':        self.year_fraction,
            'two_years':    self.two_year_fraction,
        }

        kdes = dict()
        for period_name, period_dict in period_fractions.items():
            all_fractions = []
            for fraction, items in period_dict.items():
                all_fractions.extend([fraction] * len(items))
            kde_xs, kde_ys = plot_kde_modulo(all_fractions, modulo=1, n_samples=dim)
            kdes[period_name] = (kde_xs, kde_ys)
            kdes[period_name + '_data'] = all_fractions

        self.__kdes = kdes
        return kdes

    def plot_patterns(self):
        plt.cla()
        plt.clf()
        plt.close()

        period_x_axis = {
            'days':   ('day', 'hours', 24),
            'weeks':  ('week', 'days', 7),
            # 'two_weeks':    ('fortnight', 'days', 14),
            'months': ('month', 'period', 3),  # early, mid, late
            # 'two_months':   ('two months', 'weeks', 9),
            # 'three_months': ('quarter', 'months', 3),
            # 'six_months':   ('six months', 'months', 6),
            'years':  ('year', 'months', 12),
            # 'two_years':    ('two years', 'months', 24),
        }

        _y_axis_label = 'count'
        _axis_font_size = 11
        _data_label_font_size = _axis_font_size + 2

        fig, axes = plt.subplots(nrows=len(period_x_axis))

        kdes = self.fractions()

        # sorts by x_mean
        for (period_name, (label, x_axis_label, x_axis_scale)), axis in zip(period_x_axis.items(), axes.flatten()):

            histogram_labels = None
            histogram_data = None
            if period_name == 'days':
                histogram_labels = list(range(1, 25))  # 1 - 24 (int)
                histogram_data = self.hour_of_day
            elif period_name == 'weeks':
                histogram_labels = days_of_week  # mon - sun
                histogram_data = self.day_of_week
            elif period_name == 'two_weeks':
                histogram_labels = days_of_week * 2
            elif period_name == 'months':
                histogram_labels = ['early', 'mid', 'late']
                histogram_data = self.month_part
            elif period_name == 'years':
                histogram_labels = month_names  # jan - dec
                histogram_data = self.month

            # plot kde
            kde_xs, kde_ys = kdes[period_name]

            # scale the kde curve to match the right number of ticks
            kde_xs *= x_axis_scale

            # # scale the kde curve up to match the histogram
            kde_ys *= len(kdes[period_name + '_data'])
            kde_ys /= len(histogram_labels)

            # plot histogram
            if histogram_data is not None:
                axis.hist([elem * len(histogram_labels) for elem in kdes[period_name + '_data']],
                          len(histogram_labels),
                          density=0,
                          range=(0, len(histogram_labels)),
                          color='cyan',
                          alpha=0.8,
                          label='histogram')

            # plot and fill area
            axis.plot(kde_xs, kde_ys, linewidth=1, label='kde')
            axis.fill_between(kde_xs, kde_ys, alpha=0.2)

            # labels and formatting
            axis.grid(True)
            # axis.set_title(label, fontsize=_data_label_font_size)
            # axis.set_ylabel(_y_axis_label, fontsize=_axis_font_size)
            # axis.set_xlabel(x_axis_label, fontsize=_axis_font_size)
            # for tick_label in axis.get_yticklabels():
            #     tick_label.set_fontsize(_axis_font_size - 1)
            # for tick_label in axis.get_xticklabels():
            #     tick_label.set_fontsize(_axis_font_size - 1)

            # fix y tick label to be labels not numbers
            if histogram_labels is not None:
                # print(histogram_labels, axis.get_xticks())
                # tick_labels = [histogram_labels[int(tick)] if float(tick) % 1 == 0 else ''
                #                for _, tick in enumerate(axis.get_xticks())]
                axis.set_xticks(list(range(1, len(histogram_labels) + 1)))
                axis.set_xticklabels(histogram_labels)
                axis.set_xlim(left=0, right=len(histogram_labels))

        fig.tight_layout()
        # plt.legend()

        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

    def likelihood(self, *timestamps: datetime.datetime):

        timestamps = sorted(timestamps)

        kdes = self.fractions()

        timestamps_likelihoods = []

        for timestamp in timestamps:
            timestamp_likelihood = []

            # replace timezone
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

            # time of day (hour)
            hour_of_day = timestamp.hour

            # day of week (starts at Monday)
            day_of_week = days_of_week[timestamp.weekday()]

            # nth 7-day-period of month (starts at 1)
            n = ((timestamp.day + 1) // 7) + 1
            nth_day_of_week = (n, day_of_week)

            # n-th full week of month (starts at 1, can be 0)
            full_week = (timestamp.day + 6 - timestamp.weekday()) // 7

            # day fraction (0 to less than 1)
            _start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            _end = (timestamp + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
            day_since_epoch = datetime.timedelta(seconds=timestamp.timestamp()).days  # 1970-01-02 -> 1

            idx = bisect.bisect_left(kdes['days'][0], day_fraction) % len(kdes['days'][0])
            timestamp_likelihood.append(kdes['days'][1][idx])

            # week fraction (0 to less than 1)
            week_fraction = (timestamp.weekday() + day_fraction) / 7
            week_since_epoch = (day_since_epoch + 3) // 7  # 1st Monday after epoch (1970-01-05) -> 1

            idx = bisect.bisect_left(kdes['weeks'][0], week_fraction) % len(kdes['weeks'][0])
            timestamp_likelihood.append(kdes['weeks'][1][idx])

            # fortnight fraction
            two_week_fraction = ((day_since_epoch + 3) % 14 + day_fraction) / 14
            two_week_since_epoch = week_since_epoch // 2

            idx = bisect.bisect_left(kdes['two_weeks'][0], two_week_fraction) % len(kdes['two_weeks'][0])
            timestamp_likelihood.append(kdes['two_weeks'][1][idx])

            # month fraction (0 to less than 1)
            _start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            _end = (_start + datetime.timedelta(days=35)).replace(day=1)
            month_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
            month_since_epoch = (timestamp.year - 1970) * 12 + timestamp.month - 1

            idx = bisect.bisect_left(kdes['months'][0], month_fraction) % len(kdes['months'][0])
            timestamp_likelihood.append(kdes['months'][1][idx])

            # which month, and is it early or late in the month
            month = month_names[timestamp.month - 1]

            if month_fraction < 0.33:
                month_part = 'early'
            elif month_fraction < 0.67:
                month_part = 'mid'
            else:
                assert month_fraction < 1
                month_part = 'end'

            # bimonthly fraction (0 to less than 1)
            two_month_fraction = (month_since_epoch % 2 + month_fraction) / 2

            idx = bisect.bisect_left(kdes['two_months'][0], two_month_fraction) % len(kdes['two_months'][0])
            timestamp_likelihood.append(kdes['two_months'][1][idx])

            # quarterly fraction (0 to less than 1)
            three_month_fraction = (month_since_epoch % 4 + month_fraction) / 4

            idx = bisect.bisect_left(kdes['three_months'][0], three_month_fraction) % len(kdes['three_months'][0])
            timestamp_likelihood.append(kdes['three_months'][1][idx])

            # biannual fraction (0 to less than 1)
            six_month_fraction = (month_since_epoch % 6 + month_fraction) / 6

            idx = bisect.bisect_left(kdes['six_months'][0], six_month_fraction) % len(kdes['six_months'][0])
            timestamp_likelihood.append(kdes['six_months'][1][idx])

            # year fraction (0 to less than 1)
            _start = timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            _end = _start.replace(year=timestamp.year + 1)
            year_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
            year_since_epoch = timestamp.year - 1970

            idx = bisect.bisect_left(kdes['years'][0], year_fraction) % len(kdes['years'][0])
            timestamp_likelihood.append(kdes['years'][1][idx])

            # biennial fraction (0 to less than 1)
            two_year_fraction = (year_since_epoch % 2 + year_fraction) / 2

            idx = bisect.bisect_left(kdes['two_years'][0], two_year_fraction) % len(kdes['two_years'][0])
            timestamp_likelihood.append(kdes['two_years'][1][idx])

            timestamps_likelihoods.append(timestamp_likelihood)

        # return timestamps_likelihoods
        return list(map(lambda xs: (sum(x ** 0.1 for x in xs) / len(xs)) ** 10, timestamps_likelihoods))

    def sessions(self):
        raise NotImplementedError


if __name__ == '__main__':
    timestamps = [datetime.datetime(2020, 1, 1, 12, 23),
                  datetime.datetime(2020, 1, 1, 12, 34),
                  datetime.datetime(2020, 1, 1, 12, 34),
                  datetime.datetime(2020, 1, 2, 11, 56),
                  datetime.datetime(2020, 1, 2, 11, 56),

                  datetime.datetime(2020, 2, 5, 12, 9),
                  datetime.datetime(2020, 2, 6, 13, 9),
                  datetime.datetime(2020, 2, 6, 13, 19),
                  datetime.datetime(2020, 2, 6, 13, 29),

                  datetime.datetime(2020, 3, 6, 11, 29),
                  datetime.datetime(2020, 3, 6, 12, 1),
                  datetime.datetime(2020, 3, 6, 12, 2),
                  datetime.datetime(2020, 3, 6, 12, 12),
                  datetime.datetime(2020, 3, 6, 12, 22),

                  datetime.datetime(2020, 4, 1, 12, 43),
                  datetime.datetime(2020, 4, 1, 12, 44),
                  datetime.datetime(2020, 4, 2, 11, 45),
                  datetime.datetime(2020, 4, 2, 11, 47),
                  datetime.datetime(2020, 4, 3, 13, 47),
                  datetime.datetime(2020, 4, 3, 13, 47),
                  datetime.datetime(2020, 4, 3, 13, 48),

                  datetime.datetime(2020, 5, 8, 10, 48),
                  datetime.datetime(2020, 5, 8, 10, 48),
                  datetime.datetime(2020, 5, 8, 10, 48),
                  datetime.datetime(2020, 5, 8, 10, 48),
                  datetime.datetime(2020, 5, 8, 10, 48),

                  datetime.datetime(2020, 6, 4, 12, 33),
                  datetime.datetime(2020, 6, 4, 12, 34),
                  datetime.datetime(2020, 6, 4, 12, 35),
                  datetime.datetime(2020, 6, 4, 12, 36),
                  datetime.datetime(2020, 6, 4, 12, 37),
                  datetime.datetime(2020, 6, 4, 12, 38),
                  ]

    tss = TimeStampSet()
    for timestamp in timestamps:
        tss.add(timestamp)

    test_timestamps = [datetime.datetime(2020, 5, 1, 11, 48),
                       datetime.datetime(2020, 5, 2, 11, 48),
                       datetime.datetime(2020, 5, 3, 11, 48),
                       datetime.datetime(2020, 5, 4, 11, 48),
                       datetime.datetime(2020, 5, 5, 11, 48),
                       datetime.datetime(2020, 5, 6, 11, 48),
                       datetime.datetime(2020, 5, 7, 11, 48),
                       datetime.datetime(2020, 5, 8, 11, 48),
                       datetime.datetime(2020, 5, 9, 11, 48),
                       datetime.datetime(2020, 5, 10, 11, 48),
                       datetime.datetime(2020, 5, 11, 11, 48),
                       datetime.datetime(2020, 5, 12, 11, 48),
                       datetime.datetime(2020, 5, 13, 11, 48),
                       datetime.datetime(2020, 5, 14, 11, 48),
                       datetime.datetime(2020, 5, 15, 11, 48),
                       datetime.datetime(2020, 5, 16, 11, 48),
                       datetime.datetime(2020, 5, 17, 11, 48),
                       datetime.datetime(2020, 5, 18, 11, 48),
                       datetime.datetime(2020, 5, 19, 11, 48),
                       datetime.datetime(2020, 5, 20, 11, 48),
                       datetime.datetime(2020, 5, 21, 11, 48),
                       datetime.datetime(2020, 5, 22, 11, 48),
                       datetime.datetime(2020, 5, 23, 11, 48),
                       datetime.datetime(2020, 5, 24, 11, 48),
                       datetime.datetime(2020, 5, 25, 11, 48),
                       datetime.datetime(2020, 5, 26, 11, 48),
                       datetime.datetime(2020, 5, 27, 11, 48),
                       datetime.datetime(2020, 5, 28, 11, 48),
                       datetime.datetime(2020, 5, 29, 11, 48),
                       datetime.datetime(2020, 5, 30, 11, 48),
                       ]
    for timestamp in test_timestamps:
        print(timestamp, tss.likelihood(timestamp)[0])

    tss.plot_patterns()
