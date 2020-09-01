import bisect
import datetime

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

    def add(self, timestamp: datetime.datetime):
        # replace timezone
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

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
            self.month_part.setdefault('end', []).append(timestamp_idx)

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

    def fractions(self):
        period_fractions = {
            'days':         self.day_fraction,
            'weeks':        self.week_fraction,
            'two_weeks':    self.two_week_fraction,
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
            kde_xs, kde_ys = plot_kde_modulo(all_fractions, modulo=1)
            kdes[period_name] = (kde_xs, kde_ys)

        return kdes

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

        return map(lambda xs: (sum(x ** 0.1 for x in xs) / len(xs)) ** 10, timestamps_likelihoods)

    def sessions(self):
        raise NotImplementedError
