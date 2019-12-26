# _*_ coding: utf-8 _*_

"""
Some utility functions for processing data according to business rules.

Author: Genpeng Xu
"""


def transform_channel(chan_name):
    if chan_name in {'商净', '工程常规'}:
        return '线下工程'
    elif chan_name == '常规':
        return '线下常规'
    else:
        return chan_name


def transform_project_flag(x):
    x = x.lower()
    if x in {'是', 'y'}:
        return '是'
    elif x in {'否', 'n'}:
        return '否'
    else:
        return '未知'


def remove_whitespace(df, cols=None):
    for col in cols:
        df[col] = df[col].astype(str).apply(lambda x: x.strip())
