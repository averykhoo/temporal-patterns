import bisect
import datetime
import math
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

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


@dataclass(eq=False)
class ModuloPattern:
    name: Optional[str] = None
    x_axis_labels: Optional[List[str]] = None
    x_axis_name: Optional[str] = None
    vector_dimension: int = 128
    modulo: Union[int, float] = 1

    # min, max, fractional parts
    min: float = field(default=math.inf, init=False, repr=False)
    max: float = field(default=-math.inf, init=False, repr=False)
    remainders: List[float] = field(default_factory=list, init=False, repr=False)  # in the interval [0, 1)

    # are these even needed
    _remainders: Dict[float, List] = field(default_factory=dict, init=False)  # in the interval [0, 1)
    _quotients: Dict[float, List] = field(default_factory=dict, init=False)

    # cached values
    __kde: Dict[int, Tuple[Tuple[float], Tuple[float]]] = field(default_factory=dict, init=False, repr=False)
    __vector: Tuple[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # check name
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError(self.name)
            elif not self.name:
                self.name = None

        # check x_axis_name
        if self.x_axis_name is not None:
            if not isinstance(self.x_axis_name, str):
                raise TypeError(self.x_axis_name)
            elif not self.x_axis_name:
                self.x_axis_name = None

        # check labels
        if self.x_axis_labels is not None:
            self.x_axis_labels = list(map(str, self.x_axis_labels))

        # check modulo
        if self.modulo == 0:
            raise ValueError('modulo must not be zero')

        # check vector dimension
        if not (isinstance(self.vector_dimension, int) and self.vector_dimension > 0):
            raise TypeError(f'vector dimension must be a positive integer, got {self.vector_dimension}')

    def add(self, value, item=None):
        # reset kde and vector
        self.__kde = dict()
        self.__vector = tuple()

        self.min = min(value, self.min)
        self.max = max(value, self.max)

        # add item
        quotient, remainder = divmod(value, self.modulo)
        self.remainders.append(remainder)
        self._quotients.setdefault(quotient, []).append(item)
        self._remainders.setdefault(remainder, []).append(item)

    def kde(self, dim=1000):
        if dim not in self.__kde:
            kde_xs, kde_ys = plot_kde_modulo(self.remainders, modulo=1, n_samples=dim)
            self.__kde[dim] = (tuple(kde_xs), tuple(kde_ys))
        return self.__kde[dim]

    @property
    def vector(self) -> Tuple[float]:
        if len(self.__vector) != self.vector_dimension:
            kde_xs, kde_ys = self.kde(self.vector_dimension)
            vector_length = sum(elem ** 2 for elem in kde_ys) ** 0.5
            self.__vector = tuple(elem / vector_length for elem in kde_ys)
        return self.__vector

    @property
    def n_periods(self):
        return max(0, (self.max - self.min) / self.modulo)

    def consecutive(self, min_length=2):
        out = []
        buffer = []
        for quotient in sorted(self._quotients.keys()):
            if not buffer:
                buffer.append(quotient)
            elif quotient - 1 == buffer[-1]:
                buffer.append(quotient)
            elif quotient != buffer[-1]:
                if len(buffer) >= min_length:
                    out.append(buffer)
                buffer = []
        return out

    def plot(self, axis: Optional[plt.Axes] = None, color: str = 'blue'):
        if axis is None:
            fig, ax = plt.subplots()
        else:
            fig = None
            ax = axis

        _y_axis_name = 'count'

        # plot kde
        kde_xs, kde_ys = self.kde()

        # if we have the data to plot a histogram
        if self.x_axis_labels is not None:
            # scale the kde curve horizontally, to match the right number of ticks
            kde_xs = [elem * len(self.x_axis_labels) for elem in kde_xs]

            # scale the kde curve vertically, to match the histogram height
            kde_ys = [elem * len(self.remainders) / len(self.x_axis_labels) for elem in kde_ys]

            # draw histogram
            ax.hist([elem * len(self.x_axis_labels) for elem in self.remainders],
                    len(self.x_axis_labels),
                    density=False,
                    range=(0, len(self.x_axis_labels)),
                    color=color,
                    alpha=0.5)

        # plot and fill area
        ax.plot(kde_xs, kde_ys, color=color, linewidth=1)
        ax.fill_between(kde_xs, kde_ys, color=color, alpha=0.2)

        # labels and formatting
        ax.grid(True)
        if self.name is not None:
            ax.set_title(self.name)
        ax.set_ylabel(_y_axis_name)
        if self.x_axis_name is not None:
            ax.set_xlabel(self.x_axis_name)

        # set x tick labels
        if self.x_axis_labels is not None:
            ax.set_xticks(list(range(1, len(self.x_axis_labels) + 1)))
            ax.set_xticklabels(self.x_axis_labels)

            # center each label
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')

        # set horizontal view limits
        ax.set_xlim(left=0, right=len(self.x_axis_labels) if self.x_axis_labels is not None else 1)

        if fig is not None:
            fig.tight_layout()

        return ax

    def likelihood(self, value: float):
        quotient, remainder = divmod(value, self.modulo)
        idx = bisect.bisect_left(self.kde()[0], remainder) % len(self.kde()[0])
        return self.kde()[1][idx]


def timestamp_hour_of_day_of_week_of_month(timestamp: datetime.datetime):
    # nth 7-day-period of month (starts at 1)
    n = ((timestamp.day + 1) // 7) + 1

    # n-th full week of month (starts at 1, can be 0)
    full_week = (timestamp.day + 6 - timestamp.weekday()) // 7

    # day of week (starts at Monday)
    day_of_week = days_of_week[timestamp.weekday()]

    return timestamp.hour, n, full_week, day_of_week


@lru_cache(maxsize=65536)
def timestamp_day(timestamp: datetime.datetime) -> float:
    _start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    day_fraction = (timestamp - _start).total_seconds() / 86400  # 0 to less than 1
    day_since_epoch = datetime.timedelta(seconds=timestamp.timestamp()).days  # 1970-01-02 -> 1
    return day_since_epoch + day_fraction


@lru_cache(maxsize=65536)
def timestamp_week(timestamp: datetime.datetime) -> float:
    quotient, remainder = divmod(timestamp_day(timestamp), 1)
    week_fraction = (timestamp.weekday() + remainder) / 7
    week_since_epoch = (quotient + 3) // 7  # 1st Monday after epoch (1970-01-05) -> 1
    return week_since_epoch + week_fraction


@lru_cache(maxsize=65536)
def timestamp_two_week(timestamp: datetime.datetime) -> float:
    quotient, remainder = divmod(timestamp_day(timestamp), 1)
    two_week_fraction = ((quotient + 3) % 14 + remainder) / 14
    two_week_since_epoch = (quotient + 3) // 14
    return two_week_since_epoch + two_week_fraction


@lru_cache(maxsize=65536)
def _timestamp_month(timestamp: datetime.datetime) -> float:
    _start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    _end = (_start + datetime.timedelta(days=35)).replace(day=1)
    month_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()  # 0 to less than 1
    month_since_epoch = (timestamp.year - 1970) * 12 + timestamp.month - 1
    return month_since_epoch + month_fraction


@lru_cache(maxsize=65536)
def timestamp_n_month(timestamp: datetime.datetime, n: int = 1) -> float:
    quotient, remainder = divmod(_timestamp_month(timestamp), 1)
    n_month_fraction = (quotient % n + remainder) / n
    n_month_since_epoch = quotient // n
    return n_month_since_epoch + n_month_fraction


@lru_cache(maxsize=65536)
def _timestamp_year(timestamp: datetime.datetime) -> float:
    _start = timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    _end = _start.replace(year=timestamp.year + 1)
    year_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
    year_since_epoch = timestamp.year - 1970
    return year_since_epoch + year_fraction


@lru_cache(maxsize=65536)
def timestamp_n_year(timestamp: datetime.datetime, n: int = 1) -> float:
    quotient, remainder = divmod(_timestamp_year(timestamp), 1)
    n_year_fraction = (quotient % n + remainder) / n
    n_year_since_epoch = quotient // n
    return n_year_since_epoch + n_year_fraction


class TimeStampSetV2:
    def __init__(self):
        # all the timestamps will be indexed here
        self.timestamps: List[datetime.datetime] = []

        # n-th day of month, n-th week of month
        self.hour_day: Dict[Tuple[int, str], Set[int]] = dict()
        self.n_day: Dict[Tuple[int, str], Set[int]] = dict()
        self.n_week: Dict[Tuple[int, str], Set[int]] = dict()

        # patterns
        self.day = ModuloPattern(name='',  # 'day',
                                 x_axis_labels=['1am', '2am', '3am', '4am', '5am', '6am',
                                                '7am', '8am', '9am', '10am', '11am', '12nn',
                                                '1pm', '2pm', '3pm', '4pm', '5pm', '6pm',
                                                '7pm', '8pm', '9pm', '10pm', '11pm', '12mn'],
                                 x_axis_name='')  # 'hour')
        self.week = ModuloPattern(name='',  # 'week',
                                  x_axis_labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                  x_axis_name='')  # 'day')
        self.two_week = ModuloPattern(name='',  # 'fortnight',
                                      x_axis_labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] * 2,
                                      x_axis_name='')  # 'day')
        self.month = ModuloPattern(name='',  # 'month',
                                   x_axis_labels=['early', 'mid', 'late'],
                                   x_axis_name='')  # '10-day period')
        self.two_month = ModuloPattern(name='',  # '2-month',
                                       x_axis_labels=['Odd', 'Even'],
                                       x_axis_name='')  # 'month')
        self.three_month = ModuloPattern(name='',  # 'quarter',
                                         x_axis_labels=['Jan/May/Sep', 'Feb/Jun/Oct', 'Mar/Jul/Nov', 'Apr/Aug/Dec'],
                                         x_axis_name='')  # 'month')
        self.six_month = ModuloPattern(name='',  # '6-month',
                                       x_axis_labels=['Jan/Jul', 'Feb/Aug', 'Mar/Sep',
                                                      'Apr/Oct', 'May/Nov', 'Jun/Dec'],
                                       x_axis_name='')  # 'month')
        self.year = ModuloPattern(name='',  # 'year',
                                  x_axis_labels=['Jan', 'Feb', 'Mar',
                                                 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep',
                                                 'Oct', 'Nov', 'Dec'],
                                  x_axis_name='')  # 'month')
        self.two_year = ModuloPattern(name='',  # '2-year',
                                      x_axis_labels=['Jan', 'Feb', 'Mar',
                                                     'Apr', 'May', 'Jun',
                                                     'Jul', 'Aug', 'Sep',
                                                     'Oct', 'Nov', 'Dec'] * 2,
                                      x_axis_name='')  # 'month')

    @property
    def _patterns(self):
        _patterns = [self.day,
                     self.week, self.two_week,
                     self.month, self.two_month, self.three_month, self.six_month,
                     self.year, self.two_year,
                     ]
        return [_pattern for _pattern in _patterns if _pattern.n_periods >= 3]

    def add(self, timestamp: datetime.datetime):
        # replace timezone
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

        # add to index
        timestamp_idx = len(self.timestamps)
        self.timestamps.append(timestamp)
        assert self.timestamps[timestamp_idx] is timestamp, 'race condition?'

        # count
        hour, n, full_week, day_of_week = timestamp_hour_of_day_of_week_of_month(timestamp)
        self.hour_day.setdefault((hour, day_of_week), set()).add(timestamp_idx)
        self.n_day.setdefault((n, day_of_week), set()).add(timestamp_idx)
        self.n_week.setdefault((full_week, day_of_week), set()).add(timestamp_idx)

        # modular patterns
        self.day.add(timestamp_day(timestamp))
        self.week.add(timestamp_week(timestamp))
        self.two_week.add(timestamp_two_week(timestamp))
        self.month.add(timestamp_n_month(timestamp))
        self.two_month.add(timestamp_n_month(timestamp, n=2))
        self.three_month.add(timestamp_n_month(timestamp, n=3))
        self.six_month.add(timestamp_n_month(timestamp, n=6))
        self.year.add(timestamp_n_year(timestamp))
        self.two_year.add(timestamp_n_year(timestamp, n=2))

    def consecutive(self, min_length=2):
        return {_pattern.name: _pattern.consecutive(min_length=min_length) for _pattern in self._patterns}

    def kdes(self, dim=1000):
        return {_pattern.name: _pattern.kde(dim=dim) for _pattern in self._patterns}

    def fractions(self):
        return {_pattern.name: sorted(_pattern.remainders) for _pattern in self._patterns}

    def plot(self):
        plt.cla()
        plt.clf()
        plt.close()

        fig, axes = plt.subplots(nrows=len(self._patterns))
        for _pattern, axis in zip(self._patterns, axes.flatten() if len(self._patterns) > 1 else [axes]):
            _pattern.plot(axis)

        fig.tight_layout()
        # plt.legend()

        plt.show()

        plt.clf()
        plt.cla()
        plt.close()

    def likelihood(self, *timestamps: datetime.datetime):
        timestamps = sorted(timestamps)

        timestamps_likelihoods = []
        for timestamp in timestamps:
            # replace timezone
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

            # modular patterns
            timestamps_likelihoods.append([
                self.day.likelihood(timestamp_day(timestamp)),
                self.week.likelihood(timestamp_week(timestamp)),
                self.two_week.likelihood(timestamp_two_week(timestamp)),
                self.month.likelihood(timestamp_n_month(timestamp)),
                self.two_month.likelihood(timestamp_n_month(timestamp, n=2)),
                self.three_month.likelihood(timestamp_n_month(timestamp, n=3)),
                self.six_month.likelihood(timestamp_n_month(timestamp, n=6)),
                self.year.likelihood(timestamp_n_year(timestamp)),
                self.two_year.likelihood(timestamp_n_year(timestamp, n=2)),
            ])

        # return timestamps_likelihoods
        return list(map(lambda xs: (sum(x ** 0.1 for x in xs) / len(xs)) ** 10, timestamps_likelihoods))


if __name__ == '__main__':
    time_stamps = [datetime.datetime(2020, 1, 1, 12, 23),
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

                   datetime.datetime(2020, 7, 2, 10, 23),
                   datetime.datetime(2020, 7, 2, 10, 24),
                   datetime.datetime(2020, 7, 2, 10, 25),
                   datetime.datetime(2020, 7, 2, 11, 26),
                   datetime.datetime(2020, 7, 2, 11, 27),
                   datetime.datetime(2020, 7, 2, 11, 28),
                   ]

    tss = TimeStampSetV2()
    for time_stamp in time_stamps:
        tss.add(time_stamp)

    test_time_stamps = [datetime.datetime(2020, 5, 1, 11, 48),
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
    for time_stamp in test_time_stamps:
        print(time_stamp, tss.likelihood(time_stamp)[0])

    tss.plot()
