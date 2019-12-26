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


def acc_v2(y_true, y_pred, use_unitize=True):
    comp = pd.DataFrame(
        np.array([y_true, y_pred]).transpose(),
        columns=['y_true', 'y_pred']
    )
    if use_unitize:
        comp = np.round(comp * 10000)
    comp = comp.loc[comp.y_true != 0]
    comp['acc'] = comp.apply(
        lambda row: row.y_pred / row.y_true if row.y_true > row.y_pred else row.y_true / row.y_pred, axis=1
    )
    return comp.acc.mean()


def mean_absolute_percent_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-7))


def add_accuracy(df, acc_col_name, act_val_col, pred_val_col):
    df[acc_col_name] = df[act_val_col] / df[pred_val_col]
    df[acc_col_name] = df[acc_col_name].apply(lambda x: 1 / x if x > 1 else x)


def _test():
    y_true = [0.0001, 0.0002, 0.0003, 0.0004]
    y_pred = [0.0002, 0.0004, 0.0006, 0.0008]
    print(acc_v2(y_true, y_pred))


if __name__ == '__main__':
    _test()
