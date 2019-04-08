# _*_ coding: utf-8 _*_

"""
Some utility functions about metrics.

Author: Genpeng Xu
"""

import numpy as np


def error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def acc(y_true, y_pred):
    return 1 - error(y_true, y_pred)


def mean_absolute_percent_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)
