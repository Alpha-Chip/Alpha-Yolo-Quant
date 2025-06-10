import numpy as np


def scale(a, k):
    return (2 ** (k - 1) - 1) / a
