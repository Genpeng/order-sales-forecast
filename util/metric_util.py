# _*_ coding: utf-8 _*_

"""
Some utility functions about metrics.

Author: Genpeng Xu
"""

import numpy as np
import pandas as pd


def error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def acc(y_true, y_pred):
    return 1 - error(y_true, y_pred)


def acc_v2(y_true, y_pred):
    comp = pd.DataFrame(
        np.array([y_true, y_pred]).transpose(),
        columns=['y_true', 'y_pred']
    )
    comp['acc'] = comp.y_pred / (comp.y_true + 1e-7)
    comp['acc'] = comp.acc.apply(lambda x: 1 / x if x > 1 else x)
    return comp.acc.mean()


def mean_absolute_percent_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-7))


def add_accuracy(df, act_val_col, pred_val_col):
    df['accuracy'] = df[act_val_col] / df[pred_val_col]
    df['accuracy'] = df.accuracy.apply(lambda x: 1 / x if x > 1 else x)
