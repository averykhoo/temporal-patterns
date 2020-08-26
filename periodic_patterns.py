import datetime

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


def analyze(timestamp: datetime.datetime):
    # replace timezone
    timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

    # n-th 7-day-period of month (starts at 1)
    n = ((timestamp.day + 1) // 7) + 1

    # day of week (starts at Monday)
    dow = days_of_week[timestamp.weekday()]

    # n-th full week of month (starts at 1, can be 0)
    full_week = (timestamp.day + 6 - timestamp.weekday()) // 7

    # day fraction (0 to less than 1)
    _start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    _end = (timestamp + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
    day_since_epoch = datetime.timedelta(seconds=timestamp.timestamp()).days  # 1970-01-02 -> 1

    # week fraction (0 to less than 1)
    week_fraction = (timestamp.weekday() + day_fraction) / 7
    week_since_epoch = (day_since_epoch + 3) // 7  # 1st Monday after epoch (1970-01-05) -> 1

    # fortnight fraction
    two_week_fraction = ((day_since_epoch + 3) % 14 + day_fraction) / 14
    two_week_since_epoch = week_since_epoch // 2

    # month fraction (0 to less than 1)
    _start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    _end = (_start + datetime.timedelta(days=35)).replace(day=1)
    month_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
    month_since_epoch = (timestamp.year - 1970) * 12 + timestamp.month - 1

    # bimonthly fraction (0 to less than 1)
    two_month_fraction = (month_since_epoch % 2 + month_fraction) / 2
    two_month_since_epoch = month_since_epoch // 2

    # quarterly fraction (0 to less than 1)
    three_month_fraction = (month_since_epoch % 4 + month_fraction) / 4
    three_month_since_epoch = month_since_epoch // 4

    # biannual fraction (0 to less than 1)
    six_month_fraction = (month_since_epoch % 6 + month_fraction) / 6
    six_month_since_epoch = month_since_epoch // 6

    # year fraction (0 to less than 1)
    _start = timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    _end = _start.replace(year=timestamp.year + 1)
    year_fraction = (timestamp - _start).total_seconds() / (_end - _start).total_seconds()
    year_since_epoch = timestamp.year - 1970

    # biennial fraction (0 to less than 1)
    two_year_fraction = (year_since_epoch % 2 + year_fraction) / 2
    two_year_since_epoch = year_since_epoch // 2

    return [n,
            dow,
            full_week,

            day_fraction,
            week_fraction,
            two_week_fraction,

            month_fraction,
            two_month_fraction,
            three_month_fraction,
            six_month_fraction,

            year_fraction,
            two_year_fraction,
            ]
