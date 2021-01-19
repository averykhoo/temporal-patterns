import bisect
import datetime
import math
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from kernel_density import plot_kde_modulo


def dot_product(vec_1, vec_2):
    return sum(x * y for x, y in zip(vec_1, vec_2))


hours_of_day = ['1am', '2am', '3am', '4am', '5am', '6am',
                '7am', '8am', '9am', '10am', '11am', '12nn',
                '1pm', '2pm', '3pm', '4pm', '5pm', '6pm',
                '7pm', '8pm', '9pm', '10pm', '11pm', '12mn']
days_of_week = ['Mon',  # 'Monday',
                'Tue',  # 'Tuesday',
                'Wed',  # 'Wednesday',
                'Thu',  # 'Thursday',
                'Fri',  # 'Friday',
                'Sat',  # 'Saturday',
                'Sun',  # 'Sunday',
                ]
month_names = ['Jan',  # 'January',
               'Feb',  # 'February',
               'Mar',  # 'March',
               'Apr',  # 'April',
               'May',  # 'May,
               'Jun',  # 'June',
               'Jul',  # 'July',
               'Aug',  # 'August',
               'Sep',  # 'September',
               'Oct',  # 'October',
               'Nov',  # 'November',
               'Dec',  # 'December',
               ]
ordinals = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th']

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
class GridPattern:
    name: str
    x_axis_labels: List[str]
    y_axis_labels: List[str]
    x_axis_name: Optional[str] = None
    y_axis_name: Optional[str] = None
    min_items: int = -1

    data: Dict[Tuple[str, str], List] = field(default_factory=dict, init=False)
    __vector: Tuple[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # check name
        if not isinstance(self.name, str):
            raise TypeError(self.name)
        elif len(self.name) == 0:
            raise ValueError(self.name)

        # check x_axis_name
        if self.x_axis_name is not None:
            if not isinstance(self.x_axis_name, str):
                raise TypeError(self.x_axis_name)
            elif not self.x_axis_name:
                self.x_axis_name = None

        # check y_axis_name
        if self.y_axis_name is not None:
            if not isinstance(self.y_axis_name, str):
                raise TypeError(self.y_axis_name)
            elif not self.y_axis_name:
                self.y_axis_name = None

        # check x_axis_labels
        self.x_axis_labels = list(self.x_axis_labels)
        if len(self.x_axis_labels) == 0:
            raise ValueError(self.x_axis_labels)
        for label in self.x_axis_labels:
            if not isinstance(label, str):
                raise TypeError(label)
            elif len(label) == 0:
                raise ValueError(label)

        # check y_axis_labels
        self.y_axis_labels = list(self.y_axis_labels)
        for label in self.y_axis_labels:
            if not isinstance(label, str):
                raise TypeError(label)
            elif len(label) == 0:
                raise ValueError(label)
        if len(self.y_axis_labels) == 0:
            raise ValueError(self.y_axis_labels)

        # adapt min_items
        if self.min_items < 0:
            self.min_items = int(math.ceil((len(self.x_axis_labels) + len(self.y_axis_labels)) / 2))

        # create blank data dict
        for x in self.x_axis_labels:
            for y in self.y_axis_labels:
                self.data[x, y] = []

    def add(self, x, y, item=None):
        # check item
        if (x, y) not in self.data:
            raise ValueError((x, y, self.data.keys()))

        # reset
        self.__vector = tuple()

        # add item
        self.data[x, y].append(item)

    @property
    def vector(self) -> Tuple[float]:
        if len(self.__vector) == 0:
            out = []
            for x in self.x_axis_labels:
                for y in self.y_axis_labels:
                    out.append(float(len(self.data[x, y])))
            vector_length = sum(elem ** 2 for elem in out) ** 0.5
            self.__vector = tuple(elem / vector_length for elem in out)
        return self.__vector

    def plot(self, axis: Optional[plt.Axes] = None, figsize=(10, 10)):
        if axis is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None
            ax = axis

        # show numbers
        data_grid = [[len(self.data[x, y]) for x in self.x_axis_labels] for y in self.y_axis_labels]
        im = ax.imshow(data_grid, cmap='YlGnBu', aspect='auto')

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('count', rotation=-90, va="bottom")

        # show all ticks
        ax.set_xticks(range(len(self.x_axis_labels)))
        ax.set_yticks(range(len(self.y_axis_labels)))
        # label them with the respective list entries
        ax.set_xticklabels(self.x_axis_labels)
        ax.set_yticklabels(self.y_axis_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=True, labeltop=True, labelbottom=True)

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Turn spines off
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        # create white grid
        ax.set_xticks([x - 0.5 for x in range(len(self.x_axis_labels) + 1)], minor=True)
        ax.set_yticks([x - 0.5 for x in range(len(self.y_axis_labels) + 1)], minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        # labels and formatting
        # ax.grid(True)
        if self.name is not None:
            ax.set_title(self.name)
        if self.x_axis_name is not None:
            ax.set_xlabel(self.x_axis_name)
        if self.y_axis_name is not None:
            ax.set_ylabel(self.y_axis_name)

        # # Loop over data dimensions and create text annotations.
        # for x_idx, x in enumerate(self.x_axis_labels):
        #     for y_idx, y in enumerate(self.y_axis_labels):
        #         text = ax.text(x_idx, y_idx, str(len(self.data[x, y])), ha="center", va="center", color="w")

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        textcolors = ["black", "white"]

        # Normalize the threshold to the images color range.
        threshold = im.norm(max(len(d) for d in self.data.values())) / 2
        kw = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
        for x_idx, x in enumerate(self.x_axis_labels):
            for y_idx, y in enumerate(self.y_axis_labels):
                kw.update(color=textcolors[int(im.norm(len(self.data[x, y])) >= threshold)])
                text = im.axes.text(x_idx, y_idx, f'{len(self.data[x, y])}', **kw)
                texts.append(text)

        if fig is not None:
            fig.tight_layout()

        return ax

    def likelihood(self, x: str, y: str):
        if (x, y) not in self.data:
            raise ValueError((x, y, self.data.keys()))

        return len(self.data[x, y]) / sum(self.vector)


@dataclass(eq=False)
class ModuloPattern:
    name: str
    x_axis_labels: Optional[List[str]] = None
    x_axis_name: Optional[str] = None
    vector_dimension: int = 128
    modulo: Union[int, float] = 1

    # check if there's enough data for the pattern to be valid
    min_periods: int = 4
    min_items: int = 12

    # min, max, fractional parts
    min: float = field(default=math.inf, init=False, repr=False)
    max: float = field(default=-math.inf, init=False, repr=False)
    remainders: List[float] = field(default_factory=list, init=False, repr=False)  # in the interval [0, 1)
    _raw: Dict[float, List] = field(default_factory=dict, init=False)

    # cached values
    __kde: Dict[int, Tuple[Tuple[float], Tuple[float]]] = field(default_factory=dict, init=False, repr=False)
    __vector: Tuple[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # check name
        if not isinstance(self.name, str):
            raise TypeError(self.name)
        elif len(self.name) == 0:
            raise ValueError(self.name)

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
        self._raw.setdefault(value, []).append(item)
        quotient, remainder = divmod(value, self.modulo)
        self.remainders.append(remainder)

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
        return max(0.0, (self.max - self.min) / self.modulo)

    @property
    def is_valid(self):
        return self.n_periods >= self.min_periods and len(self.remainders) >= self.min_items

    def consecutive(self, min_length=2):
        out = []
        buffer = []
        for value in sorted(self._raw.keys()):
            quotient = value // self.modulo
            if not buffer:
                buffer.append(quotient)
            elif quotient - 1 == buffer[-1]:
                buffer.append(quotient)
            elif quotient != buffer[-1]:
                if len(buffer) >= min_length:
                    out.append(buffer)
                buffer = []
        return out

    def plot(self, axis: Optional[plt.Axes] = None, color: str = 'blue', figsize=(10, 10)):
        if axis is None:
            fig, ax = plt.subplots(figsize=figsize)
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
    n = ordinals[((timestamp.day + 1) // 7) + 1]

    # n-th full week of month (starts at 1, can be 0)
    full_week = ordinals[(timestamp.day + 6 - timestamp.weekday()) // 7]

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
    def __init__(self, *timestamps):
        # all the timestamps will be indexed here
        self.timestamps: List[datetime.datetime] = []

        # # n-th day of month, n-th week of month
        # self.hour_day: Dict[Tuple[int, str], Set[int]] = dict()
        # self.n_day: Dict[Tuple[int, str], Set[int]] = dict()
        # self.n_week: Dict[Tuple[int, str], Set[int]] = dict()

        # patterns
        self.hour_of_day = GridPattern(name='hour / day-of-week',
                                       x_axis_name='day of week',
                                       x_axis_labels=days_of_week,
                                       y_axis_name='hour',
                                       y_axis_labels=hours_of_day)
        self.nth_day_of_month = GridPattern(name='n-th day-of-week of each month',
                                            x_axis_name='day of week',
                                            x_axis_labels=days_of_week,
                                            y_axis_name='n',
                                            y_axis_labels=ordinals[1:6])
        self.full_week_of_month = GridPattern(name='n-th full week in month / day-of-week',
                                              x_axis_name='day of week',
                                              x_axis_labels=days_of_week,
                                              y_axis_name='n-th full week',
                                              y_axis_labels=ordinals[:6])

        self.day = ModuloPattern(name='day',
                                 x_axis_labels=hours_of_day,
                                 x_axis_name='hour',
                                 min_periods=0)
        self.week = ModuloPattern(name='week',
                                  x_axis_labels=days_of_week,
                                  x_axis_name='day')
        self.two_week = ModuloPattern(name='fortnight',
                                      x_axis_labels=days_of_week * 2,
                                      x_axis_name='day',
                                      min_periods=12)  # about 6 months
        self.month = ModuloPattern(name='month',
                                   x_axis_labels=['early', 'mid', 'late'],
                                   x_axis_name='10-day period')
        self.two_month = ModuloPattern(name='2-month',
                                       x_axis_labels=['Odd', 'Even'],
                                       x_axis_name='month',
                                       min_periods=6)  # 1 year
        self.three_month = ModuloPattern(name='quarter',
                                         x_axis_labels=['Jan/May/Sep', 'Feb/Jun/Oct', 'Mar/Jul/Nov', 'Apr/Aug/Dec'],
                                         x_axis_name='month')
        self.six_month = ModuloPattern(name='6-month',
                                       x_axis_labels=['Jan/Jul', 'Feb/Aug', 'Mar/Sep',
                                                      'Apr/Oct', 'May/Nov', 'Jun/Dec'],
                                       x_axis_name='month',
                                       min_items=24)
        self.year = ModuloPattern(name='year',
                                  x_axis_labels=month_names,
                                  x_axis_name='month',
                                  min_items=3 * 12)  # across 3 years
        self.two_year = ModuloPattern(name='2-year',
                                      x_axis_labels=month_names * 2,
                                      x_axis_name='month',
                                      min_items=6 * 12)  # across 6 years

        if timestamps:
            for ts in timestamps:
                self.add(ts)

    @property
    def _patterns(self):
        _patterns = [self.day,
                     self.week,  # self.two_week,
                     self.month,  # self.two_month, self.three_month, self.six_month,
                     self.year,  # self.two_year,
                     ]
        return [_pattern for _pattern in _patterns if _pattern.is_valid]

    def __len__(self):
        return len(self.timestamps)

    def __iter__(self):
        self.timestamps = sorted(self.timestamps)
        return iter(self.timestamps)

    def add(self, timestamps: Union[datetime.datetime, Iterable[datetime.datetime]]):
        # normalize format
        if isinstance(timestamps, datetime.datetime):
            _timestamps = [timestamps]
        elif isinstance(timestamps, pd.Timestamp):
            _timestamps = [timestamps.to_pydatetime()]
        else:
            _timestamps = [ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts for ts in timestamps]

        # seems abnormal
        if not all(isinstance(timestamp, datetime.datetime) for timestamp in _timestamps):
            raise TypeError(timestamps)

        # add the timestamps
        for timestamp in _timestamps:
            # replace timezone
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

            # add to index
            timestamp_idx = len(self.timestamps)
            self.timestamps.append(timestamp)
            assert self.timestamps[timestamp_idx] is timestamp, 'race condition?'

            # count
            hour, n, full_week, day_of_week = timestamp_hour_of_day_of_week_of_month(timestamp)
            self.hour_of_day.add(day_of_week, hours_of_day[hour], timestamp_idx)
            self.nth_day_of_month.add(day_of_week, n, timestamp_idx)
            self.full_week_of_month.add(day_of_week, full_week, timestamp_idx)

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

    @property
    def vectors(self):
        return {_pattern.name: _pattern.vector for _pattern in self._patterns if _pattern.is_valid}

    def sessions(self, distance=datetime.timedelta(days=1.5)):
        self.timestamps = sorted(self.timestamps)
        out = []
        buffer = []
        prev_timestamp = self.timestamps[0]
        for timestamp in self.timestamps:
            if timestamp - prev_timestamp <= distance:
                buffer.append(timestamp)
            else:
                assert len(buffer) > 0
                out.append(buffer)
                buffer = [timestamp]
            prev_timestamp = timestamp
        if buffer:
            out.append(buffer)
        return out

    def session_set(self, distance=datetime.timedelta(days=1.5)):
        return TimeStampSetV2(session[0] for session in self.sessions(distance=distance))

    def session_likelihoods(self, distance=datetime.timedelta(days=1.5)):
        sessions = self.sessions(distance=distance)
        likelihoods = []
        for session_idx in range(len(sessions)):
            _tss = TimeStampSetV2()
            _tss.add(*[ts for session in sessions[:session_idx] + sessions[session_idx + 1:] for ts in session])
            likelihoods.append(_tss.likelihood(*sessions[session_idx]))

        out = []
        for session, likelihood in zip(sessions, likelihoods):
            out.append(((sum(x ** 0.1 for x in likelihood) / len(likelihood)) ** 10, session, likelihood))

        return sorted(out)

    def consecutive(self, min_length=2):
        return {_pattern.name: _pattern.consecutive(min_length=min_length) for _pattern in self._patterns}

    def kdes(self, dim=1000):
        return {_pattern.name: _pattern.kde(dim=dim) for _pattern in self._patterns}

    def fractions(self):
        return {_pattern.name: sorted(_pattern.remainders) for _pattern in self._patterns}

    def plot(self, figsize=(10, 10), show=True, clear=True):
        if clear:
            plt.cla()
            plt.clf()
            plt.close()

        fig, axes = plt.subplots(nrows=len(self._patterns), figsize=figsize)
        for _pattern, axis in zip(self._patterns, axes.flatten() if len(self._patterns) > 1 else [axes]):
            _pattern.plot(axis)

        fig.tight_layout()
        # plt.legend()

        if show:
            plt.show()

        if clear:
            plt.cla()
            plt.clf()
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
                # self.two_week.likelihood(timestamp_two_week(timestamp)),
                self.month.likelihood(timestamp_n_month(timestamp)),
                # self.two_month.likelihood(timestamp_n_month(timestamp, n=2)),
                # self.three_month.likelihood(timestamp_n_month(timestamp, n=3)),
                # self.six_month.likelihood(timestamp_n_month(timestamp, n=6)),
                # self.year.likelihood(timestamp_n_year(timestamp)),
                # self.two_year.likelihood(timestamp_n_year(timestamp, n=2)),
            ])

        # return timestamps_likelihoods
        return list(map(lambda xs: (sum(x ** 0.1 for x in xs) / len(xs)) ** 10, timestamps_likelihoods))

    def similarity(self, other: 'TimeStampSetV2'):
        similarities = []
        other_vectors = other.vectors
        for name, vector in self.vectors.items():
            if name in other_vectors:
                similarities.append(dot_product(vector, other_vectors[name]))

        # no similar vectors
        if len(similarities) == 0:
            return 0, 0

        # L(0.1) mean of all similarities
        return (sum(x ** 0.1 for x in similarities) / len(similarities)) ** 10, len(similarities)

    def forecast(self,
                 start_date: Optional[datetime.datetime] = None,
                 end_date: Optional[datetime.datetime] = None,
                 delta=datetime.timedelta(minutes=5),
                 ):

        # day after last seen date
        if start_date is None:
            start_date = max(self.timestamps)
            start_date = start_date + datetime.timedelta(days=1)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # add one month
        if end_date is None:
            end_date = start_date + datetime.timedelta(days=35)
            end_date = end_date.replace(day=start_date.day)

        # get list of dates to check
        timestamps = []
        while start_date < end_date:
            timestamps.append(start_date)
            start_date = start_date + delta

        # get likelihoods
        likelihoods = self.likelihood(*timestamps)
        max_likelihood = max(likelihoods)

        # normalize and return
        return [(timestamp, likelihood / max_likelihood) for timestamp, likelihood in zip(timestamps, likelihoods)]

    def plot_forecast(self,
                      start_date: Optional[datetime.datetime] = None,
                      end_date: Optional[datetime.datetime] = None,
                      delta=datetime.timedelta(minutes=5),
                      threshold: Optional[float] = None,
                      show: bool = True,
                      clear: bool = True,
                      ):
        if clear:
            plt.cla()
            plt.clf()
            plt.close()

        # plot
        fig, ax = plt.subplots()
        forecast = self.forecast(start_date=start_date, end_date=end_date, delta=delta)
        if not forecast:
            return None
        xs, ys = zip(*forecast)
        ax.plot(xs, ys, lw=1)

        # adaptive threshold (10% or 0.4)
        if threshold is None:
            threshold = min(sorted(ys, reverse=True)[len(ys) // 10], 0.4)
        ax.axhline(y=threshold, linewidth=0.5, color='green')

        # fill when above threshold
        xs = []
        ys = []
        for x, y in forecast:
            if y >= threshold:
                xs.append(x)
                ys.append(y)
            elif xs:
                ax.fill_between(xs, ys, threshold, alpha=0.8, color='green', lw=0)
                ax.fill_between(xs, threshold, 0, alpha=0.4, color='green', lw=0)
                xs = []
                ys = []

        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # no y tick labels
        ax.yaxis.set_ticks([])

        ax.xaxis.grid(True, which='minor')
        ax.grid(True)

        plt.grid(b=True, color='black', linestyle='-')
        plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.2)

        # center each label
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')

        # set view limits
        ax.set_xlim(left=forecast[0][0], right=forecast[-1][0])
        ax.set_ylim(bottom=0, top=1)

        if show:
            plt.show()

        if clear:
            plt.cla()
            plt.clf()
            plt.close()
        else:
            return plt

    def plot_session_likelihoods(self,
                                 forecast_period: Optional[datetime.timedelta] = datetime.timedelta(minutes=10),
                                 show: bool = True,
                                 clear: bool = True,
                                 figsize=(10, 10),
                                 ):
        if not self.timestamps:
            return
        self.timestamps = sorted(self.timestamps)

        if clear:
            plt.cla()
            plt.clf()
            plt.close()

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        forecast = self.forecast(start_date=self.timestamps[0] - datetime.timedelta(minutes=10),
                                 end_date=self.timestamps[-1] + forecast_period)
        xs, ys = zip(*forecast)
        ax.plot(xs, ys, lw=1)

        # adaptive threshold (10% or 0.4)
        threshold = min(sorted(ys, reverse=True)[len(ys) // 10], 0.4)
        ax.axhline(y=threshold, linewidth=0.5, color='green')

        # # fill when above threshold
        # xs = []
        # ys = []
        # for x, y in forecast:
        #     if y >= threshold:
        #         xs.append(x)
        #         ys.append(y)
        #     elif xs:
        #         ax.fill_between(xs, ys, 0, alpha=0.2, color='green', lw=0)
        #         xs = []
        #         ys = []
        # if xs:
        #     ax.fill_between(xs, ys, 0, alpha=0.2, color='green', lw=0)

        # add lines for each timestamp
        xs, ys = zip(*forecast)
        idx = 0
        for timestamp in self.timestamps:
            idx = bisect.bisect_left(xs, timestamp, lo=idx)
            likelihood = ys[idx]
            if likelihood >= threshold:
                ax.plot([timestamp, timestamp], [0, likelihood], color='blue')  # more accurate than `ax.axvline`
            else:
                print(timestamp, likelihood)
                ax.plot([timestamp, timestamp], [0, likelihood], color='red', lw=2)

        ax.xaxis.set_minor_locator(mdates.DayLocator())  # must set minor before major locator
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # no y tick labels
        ax.yaxis.set_ticks([])

        ax.xaxis.grid(True, which='minor')
        ax.grid(True)

        plt.grid(b=True, color='black', linestyle='-')
        plt.grid(b=True, which='minor', color='gray', linestyle='-', alpha=0.2, lw=0.5)

        # center each label
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('left')

        # set view limits
        ax.set_xlim(left=forecast[0][0], right=forecast[-1][0])
        ax.set_ylim(bottom=0, top=1)

        if show:
            plt.show()

        if clear:
            plt.cla()
            plt.clf()
            plt.close()
        else:
            return plt


if __name__ == '__main__':
    # fh
    time_stamps = [datetime.datetime(2018, 11, 29, 11, 24),
                   datetime.datetime(2018, 11, 29, 11, 24),
                   datetime.datetime(2018, 11, 29, 11, 24),
                   datetime.datetime(2018, 11, 30, 12, 9),
                   datetime.datetime(2018, 11, 30, 12, 10),
                   datetime.datetime(2018, 11, 30, 12, 10),
                   datetime.datetime(2018, 11, 30, 12, 11),
                   datetime.datetime(2018, 11, 30, 12, 11),
                   datetime.datetime(2018, 11, 30, 12, 11),

                   datetime.datetime(2018, 12, 13, 3, 53),
                   datetime.datetime(2018, 12, 25, 17, 56),

                   datetime.datetime(2019, 1, 1, 11, 15),
                   datetime.datetime(2019, 1, 1, 11, 15),
                   datetime.datetime(2019, 1, 7, 13, 5),
                   datetime.datetime(2019, 1, 7, 13, 6),
                   datetime.datetime(2019, 1, 7, 14, 57),
                   datetime.datetime(2019, 1, 14, 11, 19),
                   datetime.datetime(2019, 1, 14, 11, 19),
                   datetime.datetime(2019, 1, 14, 11, 19),
                   datetime.datetime(2019, 1, 18, 8, 40),
                   datetime.datetime(2019, 1, 18, 8, 40),
                   datetime.datetime(2019, 1, 18, 8, 40),
                   datetime.datetime(2019, 1, 18, 8, 48),
                   datetime.datetime(2019, 1, 28, 15, 56),
                   datetime.datetime(2019, 1, 28, 15, 56),
                   datetime.datetime(2019, 1, 28, 15, 56),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),
                   datetime.datetime(2019, 1, 28, 15, 57),

                   datetime.datetime(2019, 2, 27, 8, 59),

                   datetime.datetime(2019, 3, 1, 11, 53),
                   datetime.datetime(2019, 3, 1, 11, 53),
                   datetime.datetime(2019, 3, 10, 20, 14),
                   datetime.datetime(2019, 3, 10, 20, 15),
                   datetime.datetime(2019, 3, 10, 20, 15),
                   datetime.datetime(2019, 3, 10, 20, 15),
                   datetime.datetime(2019, 3, 10, 20, 15),
                   datetime.datetime(2019, 3, 10, 20, 15),
                   datetime.datetime(2019, 3, 15, 17, 27),
                   datetime.datetime(2019, 3, 15, 17, 27),
                   datetime.datetime(2019, 3, 15, 17, 27),
                   datetime.datetime(2019, 3, 31, 17, 45),
                   datetime.datetime(2019, 3, 31, 17, 45),
                   datetime.datetime(2019, 3, 31, 17, 45),
                   datetime.datetime(2019, 3, 31, 17, 45),
                   datetime.datetime(2019, 3, 31, 17, 45),

                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 10, 16, 7),
                   datetime.datetime(2019, 4, 25, 11, 47),
                   datetime.datetime(2019, 4, 28, 11, 53),
                   datetime.datetime(2019, 4, 28, 11, 53),

                   datetime.datetime(2019, 5, 7, 10, 16),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 7, 10, 17),
                   datetime.datetime(2019, 5, 31, 9, 59),

                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),
                   datetime.datetime(2019, 6, 4, 22, 30),

                   datetime.datetime(2019, 7, 4, 12, 31),
                   datetime.datetime(2019, 7, 4, 12, 31),
                   datetime.datetime(2019, 7, 4, 12, 31),
                   datetime.datetime(2019, 7, 4, 12, 31),
                   datetime.datetime(2019, 7, 4, 12, 32),
                   datetime.datetime(2019, 7, 4, 12, 32),
                   datetime.datetime(2019, 7, 8, 18, 53),

                   datetime.datetime(2019, 8, 5, 11, 34),
                   datetime.datetime(2019, 8, 7, 11, 43),
                   datetime.datetime(2019, 8, 12, 14, 54),
                   datetime.datetime(2019, 8, 16, 19, 40),
                   datetime.datetime(2019, 8, 16, 19, 40),
                   datetime.datetime(2019, 8, 19, 21, 43),
                   datetime.datetime(2019, 8, 19, 21, 43),

                   datetime.datetime(2019, 9, 4, 16, 57),
                   datetime.datetime(2019, 9, 9, 12, 34),
                   datetime.datetime(2019, 9, 9, 12, 34),
                   datetime.datetime(2019, 9, 9, 12, 34),
                   datetime.datetime(2019, 9, 9, 12, 34),
                   datetime.datetime(2019, 9, 9, 12, 34),
                   datetime.datetime(2019, 9, 9, 12, 34),

                   datetime.datetime(2019, 10, 8, 12, 15),
                   datetime.datetime(2019, 10, 8, 12, 18),
                   datetime.datetime(2019, 10, 8, 12, 18),
                   datetime.datetime(2019, 10, 8, 12, 18),
                   datetime.datetime(2019, 10, 8, 12, 19),
                   datetime.datetime(2019, 10, 8, 12, 19),
                   datetime.datetime(2019, 10, 8, 12, 19),
                   datetime.datetime(2019, 10, 8, 12, 19),

                   datetime.datetime(2019, 11, 4, 22, 13),
                   datetime.datetime(2019, 11, 4, 22, 13),
                   datetime.datetime(2019, 11, 4, 22, 13),
                   datetime.datetime(2019, 11, 4, 22, 13),
                   datetime.datetime(2019, 11, 4, 22, 14),
                   datetime.datetime(2019, 11, 4, 22, 14),
                   datetime.datetime(2019, 11, 4, 22, 14),
                   datetime.datetime(2019, 11, 4, 22, 14),
                   datetime.datetime(2019, 11, 4, 22, 14),
                   datetime.datetime(2019, 11, 13, 20, 48),
                   datetime.datetime(2019, 11, 18, 22, 37),

                   datetime.datetime(2019, 12, 3, 13, 11),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 33),
                   datetime.datetime(2019, 12, 4, 16, 34),
                   datetime.datetime(2019, 12, 4, 16, 34),
                   datetime.datetime(2019, 12, 4, 16, 34),
                   datetime.datetime(2019, 12, 14, 14, 33),
                   datetime.datetime(2019, 12, 16, 11, 59),

                   datetime.datetime(2020, 1, 7, 23, 20),
                   datetime.datetime(2020, 1, 7, 23, 20),
                   datetime.datetime(2020, 1, 7, 23, 20),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 7, 23, 21),
                   datetime.datetime(2020, 1, 14, 10, 33),
                   datetime.datetime(2020, 1, 17, 17, 47),
                   datetime.datetime(2020, 1, 17, 17, 47),
                   datetime.datetime(2020, 1, 17, 17, 47),
                   datetime.datetime(2020, 1, 17, 17, 47),
                   datetime.datetime(2020, 1, 17, 17, 47),

                   datetime.datetime(2020, 2, 14, 10, 27),
                   datetime.datetime(2020, 2, 14, 10, 28),
                   datetime.datetime(2020, 2, 14, 10, 28),
                   datetime.datetime(2020, 2, 14, 10, 28),
                   datetime.datetime(2020, 2, 14, 10, 29),
                   datetime.datetime(2020, 2, 14, 10, 29),
                   datetime.datetime(2020, 2, 14, 10, 30),

                   datetime.datetime(2020, 3, 3, 14, 13),
                   datetime.datetime(2020, 3, 9, 12, 16),
                   datetime.datetime(2020, 3, 9, 12, 16),
                   datetime.datetime(2020, 3, 10, 18, 8),
                   datetime.datetime(2020, 3, 13, 16, 3),
                   datetime.datetime(2020, 3, 13, 21, 21),
                   datetime.datetime(2020, 3, 13, 21, 21),
                   datetime.datetime(2020, 3, 13, 21, 22),
                   datetime.datetime(2020, 3, 13, 21, 23),
                   datetime.datetime(2020, 3, 17, 15, 6),
                   datetime.datetime(2020, 3, 17, 20, 47),

                   datetime.datetime(2020, 4, 1, 9, 59),
                   datetime.datetime(2020, 4, 3, 16, 42),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 3, 16, 43),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),
                   datetime.datetime(2020, 4, 29, 21, 24),

                   datetime.datetime(2020, 6, 2, 16, 10),
                   datetime.datetime(2020, 6, 2, 16, 10),
                   datetime.datetime(2020, 6, 3, 14, 44),
                   datetime.datetime(2020, 6, 3, 14, 44),
                   datetime.datetime(2020, 6, 3, 14, 44),
                   datetime.datetime(2020, 6, 3, 14, 44),
                   datetime.datetime(2020, 6, 3, 14, 45),
                   datetime.datetime(2020, 6, 3, 14, 45),
                   datetime.datetime(2020, 6, 3, 14, 45),
                   datetime.datetime(2020, 6, 11, 16, 50),
                   datetime.datetime(2020, 6, 30, 22, 44),

                   datetime.datetime(2020, 7, 2, 19, 3),
                   datetime.datetime(2020, 7, 2, 19, 3),
                   datetime.datetime(2020, 7, 2, 19, 4),
                   datetime.datetime(2020, 7, 2, 19, 4),

                   datetime.datetime(2020, 8, 4, 7, 34),
                   datetime.datetime(2020, 8, 6, 16, 2),
                   datetime.datetime(2020, 8, 6, 16, 2),
                   datetime.datetime(2020, 8, 6, 16, 2),
                   datetime.datetime(2020, 8, 6, 16, 2),
                   ]

    tss = TimeStampSetV2()
    tss.add(*time_stamps)

    # print(len(tss.sessions()))
    #
    # tss2 = TimeStampSetV2()
    # tss2.add(*[session[0] for session in tss.sessions()])
    #
    tss.hour_of_day.plot()
    plt.show()
    tss.nth_day_of_month.plot()
    plt.show()
    tss.full_week_of_month.plot()
    plt.show()
    #
    # tss.plot_session_likelihoods(forecast_period=datetime.timedelta(days=45))
    # pprint(tss.session_likelihoods())

    tss.plot()

    # tss.plot_forecast(datetime.datetime(2020, 9, 1), datetime.datetime(2020, 12, 31))
    # tss.plot_forecast(datetime.datetime(2020, 12, 21), datetime.datetime(2021, 1, 8))

    #
    # test_time_stamps = [datetime.datetime(2020, 5, 1, 9, 30),
    #                     datetime.datetime(2020, 5, 2, 9, 31),
    #                     datetime.datetime(2020, 5, 3, 9, 32),
    #                     datetime.datetime(2020, 5, 4, 9, 33),
    #                     datetime.datetime(2020, 5, 5, 9, 34),
    #                     datetime.datetime(2020, 5, 6, 10, 35),
    #                     datetime.datetime(2020, 5, 7, 10, 36),
    #                     datetime.datetime(2020, 5, 8, 10, 37),
    #                     datetime.datetime(2020, 5, 9, 10, 38),
    #                     datetime.datetime(2020, 5, 10, 10, 39),
    #                     datetime.datetime(2020, 5, 11, 11, 40),
    #                     datetime.datetime(2020, 5, 12, 11, 41),
    #                     datetime.datetime(2020, 5, 13, 11, 42),
    #                     datetime.datetime(2020, 5, 14, 11, 43),
    #                     datetime.datetime(2020, 5, 15, 11, 44),
    #                     datetime.datetime(2020, 5, 16, 12, 45),
    #                     datetime.datetime(2020, 5, 17, 12, 46),
    #                     datetime.datetime(2020, 5, 18, 12, 47),
    #                     datetime.datetime(2020, 5, 19, 12, 48),
    #                     datetime.datetime(2020, 5, 20, 12, 49),
    #                     datetime.datetime(2020, 5, 21, 13, 50),
    #                     datetime.datetime(2020, 5, 22, 13, 51),
    #                     datetime.datetime(2020, 5, 23, 13, 52),
    #                     datetime.datetime(2020, 5, 24, 13, 53),
    #                     datetime.datetime(2020, 5, 25, 13, 54),
    #                     datetime.datetime(2020, 5, 26, 14, 55),
    #                     datetime.datetime(2020, 5, 27, 14, 56),
    #                     datetime.datetime(2020, 5, 28, 14, 57),
    #                     datetime.datetime(2020, 5, 29, 14, 58),
    #                     datetime.datetime(2020, 5, 30, 14, 59),
    #                     ]
    # for time_stamp in test_time_stamps:
    #     print(time_stamp, tss.likelihood(time_stamp)[0])
    #
    # tss.plot()
    #
    # tss1 = TimeStampSetV2()
    # tss2 = TimeStampSetV2()
    # tss1.add(*test_time_stamps[::2])
    # tss2.add(*test_time_stamps[1::2])
    # print('\n')
    # print(tss1.similarity(tss2))
    #
    # tss1 = TimeStampSetV2()
    # tss2 = TimeStampSetV2()
    # tss1.add(*time_stamps[::2])
    # tss2.add(*time_stamps[1::2])
    # print('\n')
    # print(tss1.similarity(tss2))
    #
    # tss1 = TimeStampSetV2()
    # tss1.add(*test_time_stamps[:len(test_time_stamps) // 2])
    # tss2 = TimeStampSetV2()
    # tss2.add(*test_time_stamps[len(test_time_stamps) // 2:])
    # print('\n')
    # print(tss1.similarity(tss2))
    #
    # tss1 = TimeStampSetV2()
    # tss1.add(*time_stamps[:len(time_stamps) // 2])
    # tss2 = TimeStampSetV2()
    # tss2.add(*time_stamps[len(time_stamps) // 2:])
    # print('\n')
    # print(tss1.similarity(tss2))
    #
    # tss1 = TimeStampSetV2()
    # tss1.add(*time_stamps)
    # tss2 = TimeStampSetV2()
    # tss2.add(*test_time_stamps)
    # print('\n')
    # print(tss1.similarity(tss2))
