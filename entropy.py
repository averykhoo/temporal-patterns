"""
Functions to calculate Shannon entropy of strings
"""
import math

from functools import lru_cache

# noinspection SpellCheckingInspection
BASE64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='

# noinspection SpellCheckingInspection
HEX_CHARS = '1234567890abcdefABCDEF'


@lru_cache(maxsize=65336)
def shannon_entropy(data: str, charset: str = BASE64_CHARS) -> float:
    """
    Compute base-2 Shannon entropy for a given string for either base64 or hex charsets
    `df = df.assign(domain_entropy=df.domain.map(lambda x: shannon_entropy(str(x))))`

    Only counts 1-grams (i.e. individual chars) because this is meant for relatively short strings
    """
    char_freq = dict.fromkeys(charset, 0)

    for char in data:
        if char in char_freq:
            char_freq[char] += 1

    entropy = 0.0
    for char in char_freq:
        if char_freq[char] > 0:
            p_x = char_freq[char] * 1.0 / len(data)
            entropy += - p_x * math.log2(p_x)

    return entropy
