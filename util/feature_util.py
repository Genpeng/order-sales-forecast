# _*_ coding: utf-8 _*_

"""
Some functions about feature engineering.

Author: Genpeng Xu
"""

import gc
import numpy as np
import pandas as pd
from datetime import date


def is_spring_festival_pre(y, m):
    if y == 2019 and m == 1:
        return 1
    elif y == 2018 and m == 1:
        return 1
    elif y == 2016 and m == 12:
        return 1
    elif y == 2016 and m == 1:
        return 1
    else:
        return 0


def is_spring_festival(y, m):
    if (y == 2019 and m == 2) or \
            (y == 2018 and m == 2) or \
            (y == 2017 and m == 1) or \
            (y == 2016 and m == 2):
        return 1
    else:
        return 0


def is_spring_festival_after(y, m):
    if y == 2019 and m == 3:
        return 1
    elif y == 2018 and m == 3:
        return 1
    elif y == 2017 and m == 2:
        return 1
    elif y == 2016 and m == 3:
        return 1
    else:
        return 0


def is_merger_time(y, m):
    if (y == 2018 and m == 6) or (y == 2018 and m == 7):
        return 1
    else:
        return 0


def is_double_eleven_pre(m):
    return 1 if m == 10 else 0


def is_double_eleven(m):
    return 1 if m == 11 else 0


def is_double_eleven_after(m):
    return 1 if m == 12 else 0


def is_six_eighteen_pre(m):
    return 1 if m == 5 else 0


def is_six_eighteen(m):
    return 1 if m == 6 else 0


def is_six_eighteen_after(m):
    return 1 if m == 7 else 0


def get_quarter(m):
    if m <= 3:
        return 1
    elif m <= 6:
        return 2
    elif m <= 9:
        return 3
    elif m <= 12:
        return 4
    else:
        raise Exception("[ERROR] The value of month is illegal!!!")


def get_pre_10_days(order, dis, inv, index, year, month, day):
    X = {}
    start_dt, end_dt = date(year, month, 1), date(year, month, day)
    # 每个品类M月前i天的提货量
    if order:
        ord_tmp = order.loc[order.order_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
        ord_tmp = ord_tmp.groupby('category')[['ord_qty']].sum()
        ord_tmp = ord_tmp.reindex(index).fillna(0)
        ord_tmp['qty'] = ord_tmp.ord_qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
        X['ord_pre_%d_days' % day] = ord_tmp.values.ravel()
    # 每个品类M月前i天的分销量
    if dis:
        dis_tmp = dis.loc[dis.dis_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
        dis_tmp = dis_tmp.groupby('category')[['dis_qty']].sum()
        dis_tmp = dis_tmp.reindex(index).fillna(0)
        dis_tmp['qty'] = dis_tmp.qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
        X['dis_pre_%d_days' % day] = dis_tmp.values.ravel()
    # 每个品类M月前x天的库存
    if inv:
        inv_tmp = inv.loc[inv.inv_date.isin(pd.date_range(end=end_dt, periods=1, freq='D'))]
        inv_tmp = inv_tmp.groupby('category')[['inv_qty']].sum()
        inv_tmp = inv_tmp.reindex(index).fillna(0)
        inv_tmp['qty'] = inv_tmp.qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
        X['inv_pre_%d_days' % day] = inv_tmp.values.ravel()
    X = pd.DataFrame(X)
    return X


def prepare_dataset(order, dis, inv, year, month, periods=1, is_train=True, name_prefix=None):
    X = {}

    # 提货的统计特征
    if order:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = order[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月提货量
            X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
            X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月提货量的和（带衰减）
            X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
            X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
            X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
            X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
            X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    if dis:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = dis[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月分销量
            X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
            X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月分销量的和（带衰减）
            X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
            X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
            X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
            X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
            X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    if inv:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = inv[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月库存量
            X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
            X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月库存量的和（带衰减）
            X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
            X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
            X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
            X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
            X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 前i个月的提货量
    # 备注：如果修改了时间长度，需要相应修改 OdiData 的 prepare_data_for_predict 中的 left_bound
    if order:
        for i in range(1, 6):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['ord_pre_%s' % i] = order[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前i个月分销量
    if dis:
        for i in range(1, 4):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['dis_pre_%s' % i] = dis[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前i个月的库存量
    if inv:
        for i in range(1, 7):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['inv_pre_%s' % i] = inv[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    X = pd.DataFrame(X)

    # Categorical feature
    X['pred_month'] = month
    X['pred_quarter'] = X.pred_month.apply(lambda x: get_quarter(x))
    X['is_spring_fest'] = X.pred_month.apply(lambda x: is_spring_festival(year, x))
    X['is_double_eleven'] = X.pred_month.apply(lambda x: is_double_eleven(x))
    X['is_merger_time'] = X.pred_month.apply(lambda x: is_merger_time(year, x))

    if is_train:
        start_dt = date(year, month, 1)
        y = order[pd.date_range(start_dt, periods=periods, freq='M')].values
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X


def prepare_dataset_v2(order, dis, inv, year, month, gap, is_train=True, name_prefix=None):
    """
    Prepare training or validation or testing set.

    Arguments:
        order : DataFrame, order quantity per month
        dis : DataFrame, distribution quantity per month
        inv : DataFrame, inventory quantity per month
        year : int, the year to start predict
        month : int, the month to start predict
        gap : int, the time interval from start time
        is_train : boolean, true if prepare training set
        name_prefix : str, the prefix of each feature name

    Returns:
        X : DataFrame, all the features
        y (optional) : ndarry, all the labels
    """
    X = {}

    # 提货的统计特征
    if order:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = order[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月提货量
            X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
            X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月提货量的和（带衰减）
            X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
            X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
            X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
            X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
            X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    if dis:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = dis[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月分销量
            X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
            X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月分销量的和（带衰减）
            X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
            X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
            X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
            X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
            X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    if inv:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = inv[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月库存量
            X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
            X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月库存量的和（带衰减）
            X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
            X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
            X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
            X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
            X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 前i个月的提货量
    if order:
        for i in range(1, 7):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['ord_pre_%s' % i] = order[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前i个月分销量
    if dis:
        for i in range(1, 7):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['dis_pre_%s' % i] = dis[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前i个月的库存量
    if inv:
        for i in range(1, 7):
            if month - i <= 0:
                start_dt = date(year - 1, month + 12 - i, 1)
            else:
                start_dt = date(year, month - i, 1)
            X['inv_pre_%s' % i] = inv[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    X = pd.DataFrame(X)

    # Categorical features
    X['pred_month'] = month + gap
    X['pred_quarter'] = X.pred_month.apply(lambda x: get_quarter(x))
    X['is_spring_fest_pre'] = X.pred_month.apply(lambda x: is_spring_festival_pre(year, x))
    X['is_spring_fest'] = X.pred_month.apply(lambda x: is_spring_festival(year, x))
    X['is_spring_fest_after'] = X.pred_month.apply(lambda x: is_spring_festival_after(year, x))
    X['is_double_eleven_pre'] = X.pred_month.apply(lambda x: is_double_eleven_pre(x))
    X['is_double_eleven'] = X.pred_month.apply(lambda x: is_double_eleven(x))
    X['is_double_eleven_after'] = X.pred_month.apply(lambda x: is_double_eleven_after(x))
    X['is_six_eighteen_pre'] = X.pred_month.apply(lambda x: is_six_eighteen_pre(x))
    X['is_six_eighteen'] = X.pred_month.apply(lambda x: is_six_eighteen(x))
    X['is_six_eighteen_after'] = X.pred_month.apply(lambda x: is_six_eighteen_after(x))
    X['is_merger_time'] = X.pred_month.apply(lambda x: is_merger_time(year, x))

    if is_train:
        start_dt = date(year, month + gap, 1)
        y = order[pd.date_range(start_dt, periods=1, freq='M')].values
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X


def prepare_training_set(order, dis, inv, category, price, months, gap):
    X_l, y_l = [], []
    for y_m in months:
        y_str, m_str = y_m.split('-')
        y, m = int(y_str), int(m_str)
        # 有添加M月前10天提货、分销、库存量
        # pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, y, m)
        # X_tmp, y_tmp = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, y, m)
        # X_tmp = pd.concat([X_tmp, pre_10_days, cates.reset_index(drop=True)], axis=1)

        # 没有添加M月前10天提货、分销、库存量
        X_tmp, y_tmp = prepare_dataset_v2(order, dis, inv, y, m, gap=gap)
        X_tmp = pd.concat(
            [X_tmp, category.reset_index(drop=True), price.reset_index(drop=True)], axis=1
        )

        X_l.append(X_tmp)
        y_l.append(y_tmp)

        del X_tmp, y_tmp
        gc.collect()
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    return X_train, y_train


def prepare_testing_set(order, dis, inv, category, price, year, month, gap):
    X_test = prepare_dataset_v2(order, None, None, year, month, gap, is_train=False)
    X_test = pd.concat(
        [X_test, category.reset_index(drop=True), price.reset_index(drop=True)], axis=1
    )
    return X_test
