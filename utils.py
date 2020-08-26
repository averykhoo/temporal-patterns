import bisect
import csv
import math
import os
import re
import warnings
from ipaddress import IPv4Address
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import tld
from fastcache import clru_cache as lru_cache
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from entropy import shannon_entropy

RE_IPV4 = re.compile(r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)')
RE_FQDN = re.compile(r'(?:[^./@,;:*<>()\[\]\\\s\x00-\x1F\x7F]+\.)+[a-zA-Z]{2,}', flags=re.U)
RE_EMAIL_CHARS = re.compile(r'"?[^+@,;:*<>()\[\]\\\s\x00-\x1F\x7F]+"?', flags=re.U)
_chars = RE_EMAIL_CHARS.pattern
RE_EMAIL = re.compile(fr'{_chars}(?:\+{_chars})?@(?:{RE_FQDN.pattern}|{RE_IPV4.pattern})', flags=re.U)


def setup_pandas_printing():
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 100000)
    pd.set_option('display.expand_frame_repr', False)


class Hostname(str):
    _domain: Optional[str] = None

    @property
    def domain(self) -> str:
        if self._domain is not None:
            return self._domain

        if len(self) == 0:
            self._domain = ''
            return ''

        tld_result = tld.get_tld(self.lower(), fail_silently=True, as_object=True, fix_protocol=True)

        if tld_result is None:
            self._domain = ''
            return ''

        if tld_result.tld in {'in-addr.arpa', 'ip6.arpa'}:
            self._domain = tld_result.tld
            return tld_result.tld

        self._domain = tld_result.fld
        return tld_result.fld

    @property
    def subdomain(self) -> str:
        if len(self.domain) > 0:
            return self[:-len(self.domain)]
        return self[:]


def log_log(counts, exclude_one=True, window_fractional=0.1):
    # filter to count > 0, optionally exclude count == 1
    if exclude_one:
        counts = sorted(_c for _c in counts if _c > 1)
    else:
        counts = sorted(_c for _c in counts if _c > 0)

    # bounds for sliding window
    window_bounds = [(math.ceil(_c * (1 - window_fractional)), math.floor(_c * (1 + window_fractional)))
                     for _c in counts]

    # count items in sliding window
    # won't fix: this is O(n log(n)), could be O(n) by walking through the list with 2 pointers
    count_counts = []
    lower_bound = 0
    upper_bound = 0
    for left, right in window_bounds:
        upper_bound = bisect.bisect_right(counts, right, lo=upper_bound)
        lower_bound = bisect.bisect_left(counts, left, lo=lower_bound, hi=upper_bound)
        count_counts.append(upper_bound - lower_bound)

    # log both axes
    log_counts = [math.log2(c) for c in counts]
    log_count_counts = [math.log2(n) for n in count_counts]

    # xs = log_counts, ys = log_n_within_window
    return log_counts, log_count_counts


def linear_least_squares(xs, ys):
    xs = np.array(xs)
    ys = np.array(ys)

    linreg = LinearRegression()
    linreg.fit(xs.reshape(-1, 1), ys)

    y_pred = linreg.predict(xs.reshape(-1, 1))

    # slope, intercept (y = m x + c)
    m = linreg.coef_[0]
    c = linreg.intercept_

    # r squared score
    r2 = r2_score(ys, y_pred)

    return m, c, r2


@lru_cache(maxsize=65536)
def ip_to_decimal(ip_address: str) -> int:
    """
    equivalent to:
    return IPv4Address(ip)._ip
    """
    if not isinstance(ip_address, str):
        raise TypeError(ip_address)

    try:
        octet_1, octet_2, octet_3, octet_4 = map(int, ip_address.split('.'))
    except ValueError:
        print("IPv4 address must contain exactly 4 octets, all integers: {}".format(ip_address))
        raise

    assert 0 <= octet_1 <= 255, f'octet 1 ({octet_1}) is invalid, {ip_address}'
    assert 0 <= octet_2 <= 255, f'octet 2 ({octet_2}) is invalid, {ip_address}'
    assert 0 <= octet_3 <= 255, f'octet 3 ({octet_3}) is invalid, {ip_address}'
    assert 0 <= octet_4 <= 255, f'octet 4 ({octet_4}) is invalid, {ip_address}'

    return octet_1 * 256 ** 3 + octet_2 * 256 ** 2 + octet_3 * 256 + octet_4


@lru_cache(maxsize=65536)
def ip_is_global(ip_address: str) -> Optional[bool]:
    """
    df_http = df_http.assign(srcip_private=df_http.srcip.apply(is_private))

    Determine if IP is private or public
    :param ip_address: IP address in ipv4 or ipv6
    :return: True if IP address is private, false otherwise
    """
    # sanity check
    try:
        ip_to_decimal(ip_address)
    except ValueError:
        return None

    # parse
    _ip = IPv4Address(ip_address)

    return (_ip.is_global
            and not _ip.is_private
            and not _ip.is_link_local
            and not _ip.is_loopback
            and not _ip.is_multicast
            and not _ip.is_reserved
            and not _ip.is_unspecified)


def ip_details(ip_address):
    """
    df_http = df_http.assign(srcip_private=df_http.srcip.apply(is_private))

    Determine if IP is private or public
    :param ip_address: IP address in ipv4 or ipv6
    :type ip_address: str | unicode
    :return: True if IP address is private, false otherwise
    :rtype: bool
    """
    _ip = IPv4Address(ip_address)
    if _ip is None:
        return np.nan

    print(ip_address)
    print(f'private:     {_ip.is_private}')
    print(f'global:      {_ip.is_global}')
    print(f'link_local:  {_ip.is_link_local}')
    print(f'loopback:    {_ip.is_loopback}')
    print(f'multicast:   {_ip.is_multicast}')
    print(f'reserved:    {_ip.is_reserved}')
    print(f'unspecified: {_ip.is_unspecified}')


@lru_cache(maxsize=65536)
def get_domain(hostname: str) -> Optional[str]:
    if pd.isna(hostname) or len(hostname) == 0:
        return np.nan

    tld_result = tld.get_tld(hostname.lower(), fail_silently=True, as_object=True, fix_protocol=True)

    if tld_result is None:
        return None

    if tld_result.tld in {'in-addr.arpa',
                          'ip6.arpa',
                          }:
        return tld_result.tld

    return tld_result.fld


@lru_cache(maxsize=65536)
def get_second_level_domain(hostname: str) -> Optional[str]:
    domain = get_domain(hostname)
    if domain is None:
        return None
    subdomain = hostname[:-len(domain)].rstrip('.')
    if not subdomain:
        return domain
    return subdomain.rsplit('.', 1)[-1] + '.' + domain


@lru_cache(maxsize=65536)
def subdomain_entropy(hostname: str) -> float:
    domain = get_domain(hostname)
    if pd.isna(domain):
        return np.nan

    subdomain = hostname[:-len(domain)]
    return shannon_entropy(subdomain)


@lru_cache(maxsize=65536)
def write_csv(path, data_rows, headers=None, overwrite=False, verbose=False, allow_blank=False):
    path = os.path.abspath(path)
    n_columns = None

    if os.path.exists(path):
        if not overwrite:
            if verbose:
                print(f'csv already exists at <{path}> and overwrite is disabled, skipping...')
            else:
                warnings.warn(f'csv already exists at <{path}> and overwrite is disabled, skipping...')
            return

        if verbose:
            print(f'OVERWRITING <{path}>...')

    temp_path = f'{path}.partial'
    row_count = 0

    with open(temp_path, 'wt', encoding='utf8', newline='') as f:
        c = csv.writer(f)
        if headers is not None:
            c.writerow(headers)
            n_columns = len(headers)
        for i, row in enumerate(data_rows):
            row = list(row)
            if n_columns is None:
                n_columns = len(row)
            elif n_columns != len(row):
                warnings.warn(f'WARNING: row {i + 1} length is wrong! expected={n_columns}, actual={len(row)}')
            c.writerow(row)
            row_count += 1

    if row_count == 0 and not allow_blank:
        if verbose:
            print(f'zero lines to be written to <{path}>, skipping...')
        os.remove(temp_path)
        return

    if overwrite and os.path.exists(path):
        os.remove(path)

    assert not os.path.exists(path), f'race condition on <{path}>'
    os.rename(temp_path, path)

    if verbose:
        print(f'{row_count} rows written to <{path}> ({n_columns} columns)')

    return path


@lru_cache(maxsize=65536)
def format_bytes(num_bytes: int) -> str:
    # handle negatives
    if num_bytes < 0:
        minus = '-'
    else:
        minus = ''
    num_bytes = abs(num_bytes)

    # ±1 byte (singular form)
    if num_bytes == 1:
        return f'{minus}1 Byte'

    # determine unit
    unit = 0
    while unit < 8 and num_bytes > 999:
        num_bytes /= 1024.0
        unit += 1
    unit = ['Bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'][unit]

    # exact or float
    if num_bytes % 1:
        return f'{minus}{num_bytes:,.2f} {unit}'
    else:
        return f'{minus}{num_bytes:,.0f} {unit}'


@lru_cache(maxsize=65536)
def format_seconds(num_seconds: Union[int, float]) -> str:
    """
    string formatting
    note that the days in a month is kinda fuzzy
    kind of takes leap years into account, but as a result the years are fuzzy
    """

    # handle negatives
    if num_seconds < 0:
        minus = '-'
    else:
        minus = ''
    num_seconds = abs(num_seconds)

    # zero (not compatible with decimals below)
    if num_seconds == 0:
        return '0 seconds'

    # 1 or more seconds
    if num_seconds >= 1:
        unit = 0
        denominators = [60.0, 60.0, 24.0, 7.0, 365.25 / 84.0, 12.0, 10.0]
        while unit < 6 and num_seconds > denominators[unit] * 0.9:
            num_seconds /= denominators[unit]
            unit += 1
        unit_str = ['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years', 'decades'][unit]

        # singular form
        if num_seconds == 1:
            unit_str = unit_str[:-1]

        # exact or float
        if num_seconds % 1:
            return f'{minus}{num_seconds:,.2f} {unit_str}'
        else:
            return f'{minus}{num_seconds:,.0f} {unit_str}'

    # fractions of a second (ms, μs, ns)
    else:
        unit = 0
        while unit < 3 and num_seconds < 0.9:
            num_seconds *= 1000
            unit += 1
        unit = ['seconds', 'milliseconds', 'microseconds', 'nanoseconds'][unit]

        # singular form
        if num_seconds == 1:
            unit = unit[:-1]

        # exact or float
        if num_seconds % 1 and num_seconds > 1:
            return f'{minus}{num_seconds:,.2f} {unit}'
        elif num_seconds % 1:
            # noinspection PyStringFormat
            num_seconds = f'{{N:,.{1 - int(math.floor(math.log10(abs(num_seconds))))}f}}'.format(N=num_seconds)
            return f'{minus}{num_seconds} {unit}'
        else:
            return f'{minus}{num_seconds:,.0f} {unit}'


@lru_cache(maxsize=65536)
def levenshtein_distance(s1, s2):
    """
    copied from jellyfish library
    """

    if isinstance(s1, bytes) or isinstance(s2, bytes):
        raise TypeError

    if s1 == s2:
        return 0

    n_rows = len(s1) + 1
    n_cols = len(s2) + 1

    if not s1:
        return n_cols - 1
    if not s2:
        return n_rows - 1

    curr = list(range(n_cols))
    for r in range(1, n_rows):
        prev, curr = curr, [r] + [0] * (n_cols - 1)
        for c in range(1, n_cols):
            deletion = prev[c] + 1
            insertion = curr[c - 1] + 1
            substitution = prev[c - 1] + (0 if s1[r - 1] == s2[c - 1] else 1)
            curr[c] = min(deletion, insertion, substitution)

    return curr[-1]


def sort_fqdns(fqdns: Iterable[str], unique=True, lowercase=True):
    subdomain_map = dict()
    for fqdn in sorted(fqdns, key=len):
        if lowercase:
            fqdn = fqdn.lower()
        for domain in subdomain_map:
            if fqdn.endswith(domain):
                subdomain_map[domain].append(fqdn)
                break
        else:
            subdomain_map[fqdn] = []

    out = []
    for domain in sorted(subdomain_map):
        out.append(domain)
        if unique:
            out.extend(sorted(set(subdomain_map[domain])))
        else:
            out.extend(sorted(subdomain_map[domain]))

    return out


if __name__ == '__main__':
    def email_check(email, expected, x, y='2'):

        if RE_EMAIL.findall(email):
            if y == 'multiple':
                found = ','.join(RE_EMAIL.findall(email))
            else:
                found = RE_EMAIL.findall(email)[0]
        else:
            found = None

        if (found == expected) ^ x:
            print('-' * 100)
            print(repr(email))
            print(expected.split(','))
            print(RE_EMAIL.findall(email))
            print(found == expected, x, y)
            print('-' * 100)


    print("Valid single addresses when ''multiple'' attribute is not set.")
    email_check("something@something.com", "something@something.com", True)
    email_check("someone@localhost.localdomain", "someone@localhost.localdomain", True)
    email_check("someone@127.0.0.1", "someone@127.0.0.1", True)
    email_check("a@b.b", "a@b.b", True)
    email_check("a/b@domain.com", "a/b@domain.com", True)
    email_check("{}@domain.com", "{}@domain.com", True)
    email_check("m*'!%@something.sa", "m*'!%@something.sa", True)
    email_check("tu!!7n7.ad##0!!!@company.ca", "tu!!7n7.ad##0!!!@company.ca", True)
    email_check("%@com.com", "%@com.com", True)
    email_check("!#$%&'*+/=?^_`{|}~.-@com.com", "!#$%&'*+/=?^_`{|}~.-@com.com", True)
    email_check(".wooly@example.com", ".wooly@example.com", True)
    email_check("wo..oly@example.com", "wo..oly@example.com", True)
    email_check("someone@do-ma-in.com", "someone@do-ma-in.com", True)
    email_check("somebody@example", "somebody@example", True)
    email_check("\u000Aa@p.com\u000A", "a@p.com", True)
    email_check("\u000Da@p.com\u000D", "a@p.com", True)
    email_check("a\u000A@p.com", "a@p.com", True)
    email_check("a\u000D@p.com", "a@p.com", True)
    email_check("", "", True)
    email_check(" ", "", True)
    email_check(" a@p.com", "a@p.com", True)
    email_check("a@p.com ", "a@p.com", True)
    email_check(" a@p.com ", "a@p.com", True)
    email_check("\u0020a@p.com\u0020", "a@p.com", True)
    email_check("\u0009a@p.com\u0009", "a@p.com", True)
    email_check("\u000Ca@p.com\u000C", "a@p.com", True)

    print("Invalid single addresses when ''multiple'' attribute is not set.")
    email_check("invalid:email@example.com", "invalid:email@example.com", False)
    email_check("@somewhere.com", "@somewhere.com", False)
    email_check("example.com", "example.com", False)
    email_check("@@example.com", "@@example.com", False)
    email_check("a space@example.com", "a space@example.com", False)
    email_check("something@ex..ample.com", "something@ex..ample.com", False)
    email_check("a\b@c", "a\b@c", False)
    email_check("someone@somewhere.com.", "someone@somewhere.com.", False)
    email_check("\"\"test\blah\"\"@example.com", "\"\"test\blah\"\"@example.com", False)
    email_check("\"testblah\"@example.com", "\"testblah\"@example.com", False)
    email_check("someone@somewhere.com@", "someone@somewhere.com@", False)
    email_check("someone@somewhere_com", "someone@somewhere_com", False)
    email_check("someone@some:where.com", "someone@some:where.com", False)
    email_check(".", ".", False)
    email_check("F/s/f/a@feo+re.com", "F/s/f/a@feo+re.com", False)
    email_check("some+long+email+address@some+host-weird-/looking.com",
                "some+long+email+address@some+host-weird-/looking.com", False)
    email_check("a @p.com", "a @p.com", False)
    email_check("a\u0020@p.com", "a\u0020@p.com", False)
    email_check("a\u0009@p.com", "a\u0009@p.com", False)
    email_check("a\u000B@p.com", "a\u000B@p.com", False)
    email_check("a\u000C@p.com", "a\u000C@p.com", False)
    email_check("a\u2003@p.com", "a\u2003@p.com", False)
    email_check("a\u3000@p.com", "a\u3000@p.com", False)
    email_check("ddjk-s-jk@asl-.com", "ddjk-s-jk@asl-.com", False)
    email_check("someone@do-.com", "someone@do-.com", False)
    email_check("somebody@-p.com", "somebody@-p.com", False)
    email_check("somebody@-.com", "somebody@-.com", False)

    print("Valid single addresses when ''multiple'' attribute is set.")
    email_check("something@something.com", "something@something.com", True, 'multiple')
    email_check("someone@localhost.localdomain", "someone@localhost.localdomain", True, 'multiple')
    email_check("someone@127.0.0.1", "someone@127.0.0.1", True, 'multiple')
    email_check("a@b.b", "a@b.b", True, 'multiple')
    email_check("a/b@domain.com", "a/b@domain.com", True, 'multiple')
    email_check("{}@domain.com", "{}@domain.com", True, 'multiple')
    email_check("m*'!%@something.sa", "m*'!%@something.sa", True, 'multiple')
    email_check("tu!!7n7.ad##0!!!@company.ca", "tu!!7n7.ad##0!!!@company.ca", True, 'multiple')
    email_check("%@com.com", "%@com.com", True, 'multiple')
    email_check("!#$%&'*+/=?^_`{|}~.-@com.com", "!#$%&'*+/=?^_`{|}~.-@com.com", True, 'multiple')
    email_check(".wooly@example.com", ".wooly@example.com", True, 'multiple')
    email_check("wo..oly@example.com", "wo..oly@example.com", True, 'multiple')
    email_check("someone@do-ma-in.com", "someone@do-ma-in.com", True, 'multiple')
    email_check("somebody@example", "somebody@example", True, 'multiple')
    email_check("\u0020a@p.com\u0020", "a@p.com", True, 'multiple')
    email_check("\u0009a@p.com\u0009", "a@p.com", True, 'multiple')
    email_check("\u000Aa@p.com\u000A", "a@p.com", True, 'multiple')
    email_check("\u000Ca@p.com\u000C", "a@p.com", True, 'multiple')
    email_check("\u000Da@p.com\u000D", "a@p.com", True, 'multiple')
    email_check("a\u000A@p.com", "a@p.com", True, 'multiple')
    email_check("a\u000D@p.com", "a@p.com", True, 'multiple')
    email_check("", "", True, 'multiple')
    email_check(" ", "", True, 'multiple')
    email_check(" a@p.com", "a@p.com", True, 'multiple')
    email_check("a@p.com ", "a@p.com", True, 'multiple')
    email_check(" a@p.com ", "a@p.com", True, 'multiple')

    print("Invalid single addresses when ''multiple'' attribute is set.")
    email_check("invalid:email@example.com", "invalid:email@example.com", False, 'multiple')
    email_check("@somewhere.com", "@somewhere.com", False, 'multiple')
    email_check("example.com", "example.com", False, 'multiple')
    email_check("@@example.com", "@@example.com", False, 'multiple')
    email_check("a space@example.com", "a space@example.com", False, 'multiple')
    email_check("something@ex..ample.com", "something@ex..ample.com", False, 'multiple')
    email_check("a\b@c", "a\b@c", False, 'multiple')
    email_check("someone@somewhere.com.", "someone@somewhere.com.", False, 'multiple')
    email_check("\"\"test\blah\"\"@example.com", "\"\"test\blah\"\"@example.com", False, 'multiple')
    email_check("\"testblah\"@example.com", "\"testblah\"@example.com", False, 'multiple')
    email_check("someone@somewhere.com@", "someone@somewhere.com@", False, 'multiple')
    email_check("someone@somewhere_com", "someone@somewhere_com", False, 'multiple')
    email_check("someone@some:where.com", "someone@some:where.com", False, 'multiple')
    email_check(".", ".", False, 'multiple')
    email_check("F/s/f/a@feo+re.com", "F/s/f/a@feo+re.com", False, 'multiple')
    email_check("some+long+email+address@some+host-weird-/looking.com",
                "some+long+email+address@some+host-weird-/looking.com", False, 'multiple')
    email_check("\u000Ba@p.com\u000B", "\u000Ba@p.com\u000B", False, 'multiple')
    email_check("\u2003a@p.com\u2003", "\u2003a@p.com\u2003", False, 'multiple')
    email_check("\u3000a@p.com\u3000", "\u3000a@p.com\u3000", False, 'multiple')
    email_check("a @p.com", "a @p.com", False, 'multiple')
    email_check("a\u0020@p.com", "a\u0020@p.com", False, 'multiple')
    email_check("a\u0009@p.com", "a\u0009@p.com", False, 'multiple')
    email_check("a\u000B@p.com", "a\u000B@p.com", False, 'multiple')
    email_check("a\u000C@p.com", "a\u000C@p.com", False, 'multiple')
    email_check("a\u2003@p.com", "a\u2003@p.com", False, 'multiple')
    email_check("a\u3000@p.com", "a\u3000@p.com", False, 'multiple')
    email_check("ddjk-s-jk@asl-.com", "ddjk-s-jk@asl-.com", False, 'multiple')
    email_check("someone@do-.com", "someone@do-.com", False, 'multiple')
    email_check("somebody@-p.com", "somebody@-p.com", False, 'multiple')
    email_check("somebody@-.com", "somebody@-.com", False, 'multiple')

    print("Valid 'multiple' addresses when ''multiple'' attribute is set.")
    email_check("someone@somewhere.com,john@doe.com,a@b.c,a/b@c.c,ualla@ualla.127",
                "someone@somewhere.com,john@doe.com,a@b.c,a/b@c.c,ualla@ualla.127", True, 'multiple')
    email_check("tu!!7n7.ad##0!!!@company.ca,F/s/f/a@feo-re.com,m*'@a.b",
                "tu!!7n7.ad##0!!!@company.ca,F/s/f/a@feo-re.com,m*'@a.b", True, 'multiple')
    email_check(" a@p.com,b@p.com", "a@p.com,b@p.com", True, 'multiple')
    email_check("a@p.com ,b@p.com", "a@p.com,b@p.com", True, 'multiple')
    email_check("a@p.com, b@p.com", "a@p.com,b@p.com", True, 'multiple')
    email_check("a@p.com,b@p.com ", "a@p.com,b@p.com", True, 'multiple')
    email_check("   a@p.com   ,   b@p.com   ", "a@p.com,b@p.com", True, 'multiple')
    email_check("\u0020a@p.com\u0020,\u0020b@p.com\u0020", "a@p.com,b@p.com", True, 'multiple')
    email_check("\u0009a@p.com\u0009,\u0009b@p.com\u0009", "a@p.com,b@p.com", True, 'multiple')
    email_check("\u000Aa@p.com\u000A,\u000Ab@p.com\u000A", "a@p.com,b@p.com", True, 'multiple')
    email_check("\u000Ca@p.com\u000C,\u000Cb@p.com\u000C", "a@p.com,b@p.com", True, 'multiple')
    email_check("\u000Da@p.com\u000D,\u000Db@p.com\u000D", "a@p.com,b@p.com", True, 'multiple')

    print("Invalid 'multiple' addresses when ''multiple'' attribute is set.")
    email_check("someone@somewhere.com,john@doe..com,a@b,a/b@c,ualla@ualla.127",
                "someone@somewhere.com,john@doe..com,a@b,a/b@c,ualla@ualla.127", False, 'multiple')
    email_check("some+long+email+address@some+host:weird-/looking.com,F/s/f/a@feo+re.com,,m*'@'!%",
                "some+long+email+address@some+host:weird-/looking.com,F/s/f/a@feo+re.com,,m*'@'!%", False,
                'multiple')
    email_check("   a @p.com   ,   b@p.com   ", "a @p.com,b@p.com", False, 'multiple')
    email_check("   a@p.com   ,   b @p.com   ", "a@p.com,b @p.com", False, 'multiple')
    email_check("\u000Ba@p.com\u000B,\u000Bb@p.com\u000B", "\u000Ba@p.com\u000B,\u000Bb@p.com\u000B", False,
                'multiple')
    email_check("\u2003a@p.com\u2003,\u2003b@p.com\u2003", "\u2003a@p.com\u2003,\u2003b@p.com\u2003", False,
                'multiple')
    email_check("\u3000a@p.com\u3000,\u3000b@p.com\u3000", "\u3000a@p.com\u3000,\u3000b@p.com\u3000", False,
                'multiple')
    email_check(",,", ",,", False, 'multiple')
    email_check(" ,,", ",,", False, 'multiple')
    email_check(", ,", ",,", False, 'multiple')
    email_check(",, ", ",,", False, 'multiple')
    email_check("  ,  ,  ", ",,", False, 'multiple')
    email_check("\u0020,\u0020,\u0020", ",,", False, 'multiple')
    email_check("\u0009,\u0009,\u0009", ",,", False, 'multiple')
    email_check("\u000A,\u000A,\u000A", ",,", False, 'multiple')
    email_check("\u000B,\u000B,\u000B", "\u000B,\u000B,\u000B", False, 'multiple')
    email_check("\u000C,\u000C,\u000C", ",,", False, 'multiple')
    email_check("\u000D,\u000D,\u000D", ",,", False, 'multiple')
    email_check("\u2003,\u2003,\u2003", "\u2003,\u2003,\u2003", False, 'multiple')
    email_check("\u3000,\u3000,\u3000", "\u3000,\u3000,\u3000", False, 'multiple')
