# _*_ coding: utf-8 _*_

"""
Some functions about feature engineering.

Author: Genpeng Xu
"""

import gc
import numpy as np
import pandas as pd
from datetime import date
from typing import Union, Tuple

# Own customized modules
from util.date_util import infer_month
from data_loader.customer_list import CustomerList
from data_loader.item_list import ItemList


def is_spring_festival_pre(y, m):
    if y == 2019 and m == 12:
        return 1
    elif y == 2019 and m == 1:
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
    if y == 2020 and m == 1:
        return 1
    elif y == 2019 and m == 2:
        return 1
    elif y == 2018 and m == 2:
        return 1
    elif y == 2017 and m == 1:
        return 1
    elif y == 2016 and m == 2:
        return 1
    else:
        return 0


def is_spring_festival_after(y, m):
    if y == 2020 and m == 2:
        return 1
    elif y == 2019 and m == 3:
        return 1
    elif y == 2018 and m == 3:
        return 1
    elif y == 2017 and m == 2:
        return 1
    elif y == 2016 and m == 3:
        return 1
    else:
        return 0


def is_home_decoration(m):
    return 1 if m == 3 else 0


def is_merger_time(y, m):
    if (y == 2018 and m == 6) or (y == 2018 and m == 7):
        return 1
    else:
        return 0


def is_six_eighteen_pre_2(m):
    return 1 if m == 4 else 0


def is_six_eighteen_pre_1(m):
    return 1 if m == 5 else 0


def is_six_eighteen(m):
    return 1 if m == 6 else 0


def is_six_eighteen_after(m):
    return 1 if m == 7 else 0


def is_double_eleven_pre_2(m):
    return 1 if m == 9 else 0


def is_double_eleven_pre_1(m):
    return 1 if m == 10 else 0


def is_double_eleven(m):
    return 1 if m == 11 else 0


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


def is_double_twelve(m):
    return 1 if m == 12 else 0


def get_val(df: pd.DataFrame,
            year: int,
            month: int,
            need_index: bool = False) -> Union[np.ndarray, pd.Series]:
    start_dt = date(year, month, 1)
    res = df[pd.date_range(start=start_dt, periods=1, freq='M')].iloc[:, 0]
    return res if need_index else res.values


def get_pre_vals(df: pd.DataFrame,
                 year: int, month: int,
                 periods: int = 3,
                 need_index: bool = False) -> pd.DataFrame:
    end_dt = date(year, month, 1)
    res = df[pd.date_range(end=end_dt, periods=periods, freq='M')]
    return res if need_index else res.reset_index(drop=True)


def get_pre_15_days(order, dis, inv, year, month, prefix=None):
    X = pd.DataFrame()
    if order is not None:
        X['ord_pre_15_days'] = get_val(order, year, month)
    if dis is not None:
        X['dis_pre_15_days'] = get_val(dis, year, month)
    if inv is not None:
        X['inv_pre_15_days'] = get_val(inv, year, month)
    if prefix:
        X.columns = ['%s_%s' % (prefix, c) for c in X.columns]
    return X


def modify_training_set(X_train: pd.DataFrame,
                        y_train: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
    X_train['y'] = y_train
    X_train = X_train.loc[X_train.y > 0]
    y_train = X_train['y'].values
    X_train.drop(columns=['y'], inplace=True)
    return X_train, y_train


def prepare_dataset_for_level3(order: pd.DataFrame,
                               dis: pd.DataFrame,
                               inv: pd.DataFrame,
                               year: int,
                               month: int,
                               gap: int = 0,
                               add_cate_feat: bool = True,
                               name_prefix: str = None,
                               is_train: bool = True,
                               label_data: str = 'order') -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Prepare training or validation or testing set for level 3.

    Arguments:
        order: DataFrame, order quantity per month
        dis: DataFrame, distribution quantity per month
        inv: DataFrame, inventory quantity per month
        year: int, the year to start predict
        month: int, the month to start predict
        gap: int, the time interval from start time
        add_cate_feat: bool, true if add categorical features
        name_prefix: str, the prefix of each feature name
        is_train: bool, true if prepare training set
        label_data: str, the type of data to predict

    Returns:
        X: DataFrame, all the features
        y (optional): ndarray, all the labels
    """
    X = {}
    true_pred_year, true_pred_month = infer_month(year, month, gap)

    # 提货的统计特征
    if order is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(order, year, month, periods=i)  # 前i个月提货量
            X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
            X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月提货量的和（带衰减）
            X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
            X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
            X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
            X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
            X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    if dis is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(dis, year, month, periods=i)  # 前i个月分销量
            X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
            X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月分销量的和（带衰减）
            X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
            X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
            X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
            X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
            X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    if inv is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(inv, year, month, periods=i)  # 前i个月库存量
            X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
            X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
                axis=1).values  # 前i个月库存量的和（带衰减）
            X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
            X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
            X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
            X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
            X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 提货月数特征
    if order is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(order, year, month, periods=i)
            X['has_ord_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有提货的天数
            # X['last_ord_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有提货的天数
            # X['first_ord_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有提货的天数

    # 分销月数特征
    if dis is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(dis, year, month, periods=i)
            X['has_dis_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有分销的天数
            # X['last_dis_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有分销的天数
            # X['first_dis_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有分销的天数

    # 库存月数特征
    if inv is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(inv, year, month, periods=i)
            X['has_inv_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有库存的天数
            # X['last_inv_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有库存的天数
            # X['first_inv_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有库存的天数

    # 前i个月的提货量
    if order is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['ord_pre_%s' % i] = get_val(order, y_tmp, m_tmp)

    # 前i个月分销量
    if dis is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['dis_pre_%s' % i] = get_val(dis, y_tmp, m_tmp)

    # 前i个月的库存量
    if inv is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['inv_pre_%s' % i] = get_val(order, y_tmp, m_tmp)

    # 历史同期前后3个月的提货
    # if order is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['ord_his_%s' % i] = get_val(order, y_curr, m_curr)

    # 历史同期前后3个月的分销
    # if dis is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['dis_his_%s' % i] = get_val(dis, y_curr, m_curr)

    # 历史同期前后3个月的库存
    # if inv is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['inv_his_%s' % i] = get_val(inv, y_curr, m_curr)

    X = pd.DataFrame(X)

    if add_cate_feat:
        X['pred_month'] = true_pred_month
        X['pred_quarter'] = X.pred_month.apply(lambda x: get_quarter(x))

        X['is_spring_fest_pre'] = X.pred_month.apply(lambda x: is_spring_festival_pre(true_pred_year, x))
        X['is_spring_fest'] = X.pred_month.apply(lambda x: is_spring_festival(true_pred_year, x))
        X['is_spring_fest_after'] = X.pred_month.apply(lambda x: is_spring_festival_after(true_pred_year, x))
        X['is_home_decoration'] = X.pred_month.apply(lambda x: is_home_decoration(x))
        X['is_six_eighteen_pre_2'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_2(x))
        X['is_six_eighteen_pre_1'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_1(x))
        X['is_six_eighteen'] = X.pred_month.apply(lambda x: is_six_eighteen(x))
        X['is_six_eighteen_after'] = X.pred_month.apply(lambda x: is_six_eighteen_after(x))
        X['is_merger_time'] = X.pred_month.apply(lambda x: is_merger_time(true_pred_year, x))
        X['is_double_eleven_pre_2'] = X.pred_month.apply(lambda x: is_double_eleven_pre_2(x))
        X['is_double_eleven_pre_1'] = X.pred_month.apply(lambda x: is_double_eleven_pre_1(x))
        X['is_double_eleven'] = X.pred_month.apply(lambda x: is_double_eleven(x))
        X['is_double_twelve'] = X.pred_month.apply(lambda x: is_double_twelve(x))

        if order is not None:
            tmp = order.copy()
        elif dis is not None:
            tmp = dis.copy()
        else:
            tmp = inv.copy()
        index_df = tmp.reset_index()
        index_df.drop(tmp.columns, axis=1, inplace=True)
        column_names = set(index_df.columns)
        del tmp
        gc.collect()

        if 'customer_code' in column_names:
            customer_list = CustomerList()
            X['is_white_cus'] = index_df.customer_code.apply(
                lambda x: 1 if customer_list.is_white_customer(x) else 0).values
        if 'item_code' in column_names:
            item_list = ItemList(year, month)
            X['is_white_items'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_white_items(x) else 0).values
            X['is_scheduled_delisting'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_scheduled_delisting_items(x) else 0).values
            X['is_delisting'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_delisting_items(x) else 0).values
            X['is_new'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_new_items(x) else 0).values
            X['is_curr_new'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_curr_new_items(x) else 0).values
            X['is_618_main'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_six_eighteen_main_items(x) else 0).values
            X['is_1111_main'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_double_eleven_main_items(x) else 0).values

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    if is_train:
        if label_data == 'order':
            y = get_val(order, true_pred_year, true_pred_month)
        elif label_data == 'dis':
            y = get_val(dis, true_pred_year, true_pred_month)
        elif label_data == 'inv':
            y = get_val(inv, true_pred_year, true_pred_month)
        else:
            raise Exception("[INFO] Illegal label data name!!!")
        return X, y
    else:
        return X


def prepare_training_set_for_level3(order_cus_sku_month, dis_cus_sku_month, inv_cus_sku_month,
                                    order_cus_sku_month_pre15, dis_cus_sku_month_pre15, inv_cus_sku_month_pre15,
                                    order_cus_cate1_month, dis_cus_cate1_month, inv_cus_cate1_month,
                                    order_cus_cate2_month, dis_cus_cate2_month, inv_cus_cate2_month,
                                    order_cus_chan_month, dis_cus_chan_month, inv_cus_chan_month,
                                    order_cus_sales_chan_month, dis_cus_sales_chan_month, inv_cus_sales_chan_month,
                                    order_sku_month, dis_sku_month, inv_sku_month,
                                    order_cate1_month, dis_cate1_month, inv_cate1_month,
                                    order_cate2_month, dis_cate2_month, inv_cate2_month,
                                    customer_info, sku_info, months, gap, label_data):
    X_l, y_l = [], []
    for y_m in months:
        y_str, m_str = y_m.split('-')
        y, m = int(y_str), int(m_str)

        X_tmp, y_tmp = prepare_dataset_for_level3(order_cus_sku_month,
                                                  dis_cus_sku_month,
                                                  inv_cus_sku_month, y, m, gap, label_data=label_data)
        X_tmp1 = get_pre_15_days(order_cus_sku_month_pre15, dis_cus_sku_month_pre15, inv_cus_sku_month_pre15, y, m)
        X_tmp2 = prepare_dataset_for_level3(order_cus_cate1_month,
                                            dis_cus_cate1_month,
                                            inv_cus_cate1_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cus_cate1')
        X_tmp3 = prepare_dataset_for_level3(order_cus_cate2_month,
                                            dis_cus_cate2_month,
                                            inv_cus_cate2_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cus_cate2')
        X_tmp4 = prepare_dataset_for_level3(order_cus_chan_month,
                                            dis_cus_chan_month,
                                            inv_cus_chan_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cus_chan')
        X_tmp5 = prepare_dataset_for_level3(order_cus_sales_chan_month,
                                            dis_cus_sales_chan_month,
                                            inv_cus_sales_chan_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cus_sales_chan')
        X_tmp6 = prepare_dataset_for_level3(order_sku_month,
                                            dis_sku_month,
                                            inv_sku_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='sku')
        X_tmp7 = prepare_dataset_for_level3(order_cate1_month,
                                            dis_cate1_month,
                                            inv_cate1_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cate1')
        X_tmp8 = prepare_dataset_for_level3(order_cate2_month,
                                            dis_cate2_month,
                                            inv_cate2_month,
                                            y, m, gap, add_cate_feat=False,
                                            is_train=False, name_prefix='cate2')
        X_tmp = pd.concat(
            [X_tmp, X_tmp1, X_tmp2, X_tmp3, X_tmp4, X_tmp5, X_tmp6, X_tmp7, X_tmp8,
             customer_info.reset_index(drop=True), sku_info.reset_index(drop=True)], axis=1
        )

        X_l.append(X_tmp)
        y_l.append(y_tmp)
        del X_tmp, y_tmp, X_tmp2, X_tmp3, X_tmp4, X_tmp5, X_tmp6, X_tmp7, X_tmp8
        gc.collect()

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    return X_train, y_train


def prepare_val_set_for_level3(order_cus_sku_month, dis_cus_sku_month, inv_cus_sku_month,
                               order_cus_sku_month_pre15, dis_cus_sku_month_pre15, inv_cus_sku_month_pre15,
                               order_cus_cate1_month, dis_cus_cate1_month, inv_cus_cate1_month,
                               order_cus_cate2_month, dis_cus_cate2_month, inv_cus_cate2_month,
                               order_cus_chan_month, dis_cus_chan_month, inv_cus_chan_month,
                               order_cus_sales_chan_month, dis_cus_sales_chan_month, inv_cus_sales_chan_month,
                               order_sku_month, dis_sku_month, inv_sku_month,
                               order_cate1_month, dis_cate1_month, inv_cate1_month,
                               order_cate2_month, dis_cate2_month, inv_cate2_month,
                               customer_info, sku_info, year, month, gap, label_data):
    X_val, y_val = prepare_dataset_for_level3(order_cus_sku_month,
                                              dis_cus_sku_month,
                                              inv_cus_sku_month,
                                              year, month, gap, label_data=label_data)
    X_val1 = get_pre_15_days(order_cus_sku_month_pre15,
                             dis_cus_sku_month_pre15,
                             inv_cus_sku_month_pre15,
                             year, month)
    X_val2 = prepare_dataset_for_level3(order_cus_cate1_month,
                                        dis_cus_cate1_month,
                                        inv_cus_cate1_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cus_cate1')
    X_val3 = prepare_dataset_for_level3(order_cus_cate2_month,
                                        dis_cus_cate2_month,
                                        inv_cus_cate2_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cus_cate2')
    X_val4 = prepare_dataset_for_level3(order_cus_chan_month,
                                        dis_cus_chan_month,
                                        inv_cus_chan_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cus_chan')
    X_val5 = prepare_dataset_for_level3(order_cus_sales_chan_month,
                                        dis_cus_sales_chan_month,
                                        inv_cus_sales_chan_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cus_sales_chan')
    X_val6 = prepare_dataset_for_level3(order_sku_month,
                                        dis_sku_month,
                                        inv_sku_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='sku')
    X_val7 = prepare_dataset_for_level3(order_cate1_month,
                                        dis_cate1_month,
                                        inv_cate1_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cate1')
    X_val8 = prepare_dataset_for_level3(order_cate2_month,
                                        dis_cate2_month,
                                        inv_cate2_month,
                                        year, month, gap, add_cate_feat=False,
                                        is_train=False, name_prefix='cate2')
    X_val = pd.concat(
        [X_val, X_val1, X_val2, X_val3, X_val4, X_val5, X_val6, X_val7, X_val8,
         customer_info.reset_index(drop=True), sku_info.reset_index(drop=True)], axis=1
    )
    del X_val2, X_val3, X_val4, X_val5, X_val6, X_val7, X_val8
    gc.collect()
    return X_val, y_val


def prepare_testing_set_for_level3(order_cus_sku_month, dis_cus_sku_month, inv_cus_sku_month,
                                   order_cus_sku_month_pre15, dis_cus_sku_month_pre15, inv_cus_sku_month_pre15,
                                   order_cus_cate1_month, dis_cus_cate1_month, inv_cus_cate1_month,
                                   order_cus_cate2_month, dis_cus_cate2_month, inv_cus_cate2_month,
                                   order_cus_channel_month, dis_cus_channel_month, inv_cus_channel_month,
                                   order_cus_sales_channel_month, dis_cus_sales_channel_month,
                                   inv_cus_sales_channel_month,
                                   order_sku_month, dis_sku_month, inv_sku_month,
                                   order_cate1_month, dis_cate1_month, inv_cate1_month,
                                   order_cate2_month, dis_cate2_month, inv_cate2_month,
                                   customer_info, sku_info, year, month, gap):
    X_test = prepare_dataset_for_level3(order_cus_sku_month,
                                        dis_cus_sku_month,
                                        inv_cus_sku_month,
                                        year, month, gap, is_train=False)
    X_test1 = get_pre_15_days(order_cus_sku_month_pre15, dis_cus_sku_month_pre15, inv_cus_sku_month_pre15, year, month)
    X_test2 = prepare_dataset_for_level3(order_cus_cate1_month,
                                         dis_cus_cate1_month,
                                         inv_cus_cate1_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cus_cate1')
    X_test3 = prepare_dataset_for_level3(order_cus_cate2_month,
                                         dis_cus_cate2_month,
                                         inv_cus_cate2_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cus_cate2')
    X_test4 = prepare_dataset_for_level3(order_cus_channel_month,
                                         dis_cus_channel_month,
                                         inv_cus_channel_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cus_chan')
    X_test5 = prepare_dataset_for_level3(order_cus_sales_channel_month,
                                         dis_cus_sales_channel_month,
                                         inv_cus_sales_channel_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cus_sales_chan')
    X_test6 = prepare_dataset_for_level3(order_sku_month,
                                         dis_sku_month,
                                         inv_sku_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='sku')
    X_test7 = prepare_dataset_for_level3(order_cate1_month,
                                         dis_cate1_month,
                                         inv_cate1_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cate1')
    X_test8 = prepare_dataset_for_level3(order_cate2_month,
                                         dis_cate2_month,
                                         inv_cate2_month,
                                         year, month, gap, add_cate_feat=False,
                                         is_train=False, name_prefix='cate2')
    X_test = pd.concat(
        [X_test, X_test1, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8,
         customer_info.reset_index(drop=True), sku_info.reset_index(drop=True)], axis=1
    )
    del X_test2, X_test3, X_test4, X_test5, X_test6, X_test7, X_test8
    gc.collect()
    return X_test


def prepare_dataset_for_level2(order: pd.DataFrame,
                               dis: pd.DataFrame,
                               inv: pd.DataFrame,
                               year: int,
                               month: int,
                               gap: int = 1,
                               add_cate_feat: bool = True,
                               is_train: bool = True,
                               name_prefix: str = None,
                               label_data: str = 'order') -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Prepare training or validation or testing set.

    Arguments:
        order : DataFrame, order quantity per month
        dis : DataFrame, distribution quantity per month
        inv : DataFrame, inventory quantity per month
        year : int, the year to start predict
        month : int, the month to start predict
        gap : int, the time interval from start time
        add_cate_feat: bool, true if add categorical features
        is_train : boolean, true if prepare training set
        name_prefix : str, the prefix of each feature name
        label_data: str, the type of data to predict

    Returns:
        X : DataFrame, all the features
        y (optional) : ndarry, all the labels
    """
    X = {}
    true_pred_year, true_pred_month = infer_month(year, month, gap)

    # 提货的统计特征
    # if order is not None:
    #     for i in [3, 6]:
    #         tmp = get_pre_vals(order, year, month, periods=i)  # 前i个月提货量
    #         X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
    #         X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
    #             axis=1).values  # 前i个月提货量的和（带衰减）
    #         X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
    #         X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
    #         X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
    #         X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
    #         X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    # if dis is not None:
    #     for i in [3, 6]:
    #         tmp = get_pre_vals(dis, year, month, periods=i)  # 前i个月分销量
    #         X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
    #         X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
    #             axis=1).values  # 前i个月分销量的和（带衰减）
    #         X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
    #         X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
    #         X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
    #         X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
    #         X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    # if inv is not None:
    #     for i in [3, 6]:
    #         tmp = get_pre_vals(inv, year, month, periods=i)  # 前i个月库存量
    #         X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
    #         X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
    #             axis=1).values  # 前i个月库存量的和（带衰减）
    #         X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
    #         X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
    #         X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
    #         X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
    #         X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 提货月数特征
    if order is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(order, year, month, periods=i)
            X['has_ord_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有提货的天数
            # X['last_ord_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有提货的天数
            # X['first_ord_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有提货的天数

    # 分销月数特征
    if dis is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(dis, year, month, periods=i)
            X['has_dis_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有分销的天数
            # X['last_dis_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有分销的天数
            # X['first_dis_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有分销的天数

    # 库存月数特征
    if inv is not None:
        for i in [3, 6]:
            tmp = get_pre_vals(inv, year, month, periods=i)
            X['has_inv_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有库存的天数
            # X['last_inv_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有库存的天数
            # X['first_inv_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有库存的天数

    # 前i个月的提货量
    if order is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['ord_pre_%s' % i] = get_val(order, y_tmp, m_tmp)

    # 前i个月分销量
    if dis is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['dis_pre_%s' % i] = get_val(dis, y_tmp, m_tmp)

    # 前i个月的库存量
    if inv is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            X['inv_pre_%s' % i] = get_val(order, y_tmp, m_tmp)

    # 历史同期前后3个月的提货
    # if order is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['ord_his_%s' % i] = get_val(order, y_curr, m_curr)

    # 历史同期前后3个月的分销
    # if dis is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['dis_his_%s' % i] = get_val(dis, y_curr, m_curr)

    # 历史同期前后3个月的库存
    # if inv is not None:
    #     y_his, m_his = infer_month(year, month, offset=-12)  # 历史同期
    #     for i in range(-3, 4):
    #         y_curr, m_curr = infer_month(y_his, m_his, offset=i)
    #         X['inv_his_%s' % i] = get_val(inv, y_curr, m_curr)

    X = pd.DataFrame(X)

    if add_cate_feat:
        X['pred_month'] = true_pred_month
        X['pred_quarter'] = X.pred_month.apply(lambda x: get_quarter(x))

        X['is_spring_fest_pre'] = X.pred_month.apply(lambda x: is_spring_festival_pre(true_pred_year, x))
        X['is_spring_fest'] = X.pred_month.apply(lambda x: is_spring_festival(true_pred_year, x))
        X['is_spring_fest_after'] = X.pred_month.apply(lambda x: is_spring_festival_after(true_pred_year, x))
        # X['is_home_decoration'] = X.pred_month.apply(lambda x: is_home_decoration(x))
        X['is_six_eighteen_pre_2'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_2(x))
        X['is_six_eighteen_pre_1'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_1(x))
        X['is_six_eighteen'] = X.pred_month.apply(lambda x: is_six_eighteen(x))
        X['is_six_eighteen_after'] = X.pred_month.apply(lambda x: is_six_eighteen_after(x))
        X['is_merger_time'] = X.pred_month.apply(lambda x: is_merger_time(true_pred_year, x))
        X['is_double_eleven_pre_2'] = X.pred_month.apply(lambda x: is_double_eleven_pre_2(x))
        X['is_double_eleven_pre_1'] = X.pred_month.apply(lambda x: is_double_eleven_pre_1(x))
        X['is_double_eleven'] = X.pred_month.apply(lambda x: is_double_eleven(x))
        X['is_double_twelve'] = X.pred_month.apply(lambda x: is_double_twelve(x))

        if order is not None:
            tmp = order.copy()
        elif dis is not None:
            tmp = dis.copy()
        else:
            tmp = inv.copy()
        index_df = tmp.reset_index()
        index_df.drop(tmp.columns, axis=1, inplace=True)
        column_names = set(index_df.columns)
        del tmp
        gc.collect()

        if 'customer_code' in column_names:
            customer_list = CustomerList()
            X['is_white_cus'] = index_df.customer_code.apply(
                lambda x: 1 if customer_list.is_white_customer(x) else 0).values
        if 'item_code' in column_names:
            item_list = ItemList(year, month)
            X['is_white_items'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_white_items(x) else 0).values
            X['is_scheduled_delisting'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_scheduled_delisting_items(x) else 0).values
            X['is_delisting'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_delisting_items(x) else 0).values
            X['is_new'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_new_items(x) else 0).values
            X['is_curr_new'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_curr_new_items(x) else 0).values
            X['is_618_main'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_six_eighteen_main_items(x) else 0).values
            X['is_1111_main'] = index_df.item_code.apply(
                lambda x: 1 if item_list.is_double_eleven_main_items(x) else 0).values

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    if is_train:
        if label_data == 'order':
            y = get_val(order, true_pred_year, true_pred_month)
        elif label_data == 'dis':
            y = get_val(dis, true_pred_year, true_pred_month)
        elif label_data == 'inv':
            y = get_val(inv, true_pred_year, true_pred_month)
        else:
            raise Exception("[INFO] Illegal label data name!!!")
        return X, y
    else:
        return X


def prepare_training_set_for_level2(order, dis, inv,
                                    order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                                    order_cate1_month, dis_cate1_month, inv_cate1_month,
                                    order_cate2_month, dis_cate2_month, inv_cate2_month,
                                    sku_info, months, gap, label_data):
    X_l, y_l = [], []
    for y_m in months:
        y_str, m_str = y_m.split('-')
        y, m = int(y_str), int(m_str)

        X_tmp, y_tmp = prepare_dataset_for_level2(order, dis, inv, y, m, gap, label_data=label_data)
        X_tmp1 = get_pre_15_days(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, y, m)
        X_tmp2 = prepare_dataset_for_level2(order_cate1_month,
                                            dis_cate1_month,
                                            inv_cate1_month,
                                            y, m, gap,
                                            add_cate_feat=False, is_train=False, name_prefix='cate1')
        X_tmp3 = prepare_dataset_for_level2(order_cate2_month,
                                            dis_cate2_month,
                                            inv_cate2_month,
                                            y, m, gap,
                                            add_cate_feat=False, is_train=False, name_prefix='cate2')
        X_tmp = pd.concat(
            [X_tmp, X_tmp1, X_tmp2, X_tmp3, sku_info.reset_index(drop=True)], axis=1
        )
        X_l.append(X_tmp)
        y_l.append(y_tmp)
        del X_tmp, y_tmp, X_tmp1, X_tmp2, X_tmp3
        gc.collect()
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    return X_train, y_train


def prepare_val_set_for_level2(order, dis, inv,
                               order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                               order_cate1_month, dis_cate1_month, inv_cate1_month,
                               order_cate2_month, dis_cate2_month, inv_cate2_month,
                               sku_info, year, month, gap, label_data):
    X_val, y_val = prepare_dataset_for_level2(order, dis, inv, year, month, gap, label_data=label_data)
    X_val1 = get_pre_15_days(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, year, month)
    X_val2 = prepare_dataset_for_level2(order_cate1_month,
                                        dis_cate1_month,
                                        inv_cate1_month,
                                        year, month, gap,
                                        add_cate_feat=False, is_train=False, name_prefix='cate1')
    X_val3 = prepare_dataset_for_level2(order_cate2_month,
                                        dis_cate2_month,
                                        inv_cate2_month,
                                        year, month, gap,
                                        add_cate_feat=False, is_train=False, name_prefix='cate2')
    X_val = pd.concat(
        [X_val, X_val1, X_val2, X_val3, sku_info.reset_index(drop=True)],
        axis=1
    )
    del X_val1, X_val2, X_val3
    gc.collect()
    return X_val, y_val


def prepare_testing_set_for_level2(order, dis, inv,
                                   order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                                   order_cate1_month, dis_cate1_month, inv_cate1_month,
                                   order_cate2_month, dis_cate2_month, inv_cate2_month,
                                   sku_info, year, month, gap):
    X_test = prepare_dataset_for_level2(order, dis, inv, year, month, gap, is_train=False)
    X_test1 = get_pre_15_days(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, year, month)
    X_test2 = prepare_dataset_for_level2(order_cate1_month,
                                         dis_cate1_month,
                                         inv_cate1_month,
                                         year, month, gap,
                                         add_cate_feat=False, is_train=False, name_prefix='cate1')
    X_test3 = prepare_dataset_for_level2(order_cate2_month,
                                         dis_cate2_month,
                                         inv_cate2_month,
                                         year, month, gap,
                                         add_cate_feat=False, is_train=False, name_prefix='cate2')
    X_test = pd.concat(
        [X_test, X_test1, X_test2, X_test3, sku_info.reset_index(drop=True)],
        axis=1
    )
    del X_test1, X_test2, X_test3
    gc.collect()
    return X_test
