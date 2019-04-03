# _*_ coding: utf-8 _*_

"""
Some useful utility functions about pandas.

Author: Genpeng Xu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def row_normalization(df):
    """Normalize each row of data."""
    df_bak = df.copy()
    rnames = list(df_bak.index)
    scalers = dict()
    for rn in rnames:
        scaler = MinMaxScaler().fit(df_bak.loc[rn].values.reshape(-1, 1))
        df_bak.loc[rn] = scaler.transform(df_bak.loc[rn].values.reshape(-1, 1)).ravel()
        scalers[rn] = scaler
    return df_bak, scalers


def row_restore(df_normalized, scalers):
    """Convert data back from normalized values."""
    df_bak = df_normalized.copy()
    rnames = list(df_bak.index)
    for rn in rnames:
        scaler = scalers[rn]
        df_bak.loc[rn] = scaler.inverse_transform(df_bak.loc[rn].values.reshape(-1, 1)).ravel()
    return df_bak

