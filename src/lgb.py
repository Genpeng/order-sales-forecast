# _*_ coding: utf-8 _*_

"""
Use LightGBM to predict sales of order.

Author: StrongXGP (xgp1227@gmail.com)
Date:	2018/11/20
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error

# ============================================================================================== #
# Part 1: Preprocess

sales = pd.read_csv("../data/daily-sales-2.csv", sep=',', parse_dates=['creation_date'])
items = pd.read_csv("../processed_data/item-info.csv", sep=',').set_index('item_code')
centers = pd.read_csv("../processed_data/center-info.csv", sep=',').set_index('sales_cen_wid')

sales = sales.drop_duplicates()  # 由于取数造成的重复
sales = sales[sales['creation_date'] <= '2018-08-31']  # 减少数据量
sales.set_index(['sales_cen_wid', 'customer_wid', 'item_code', 'creation_date'], inplace=True)

sales['qty'] = sales['item_qty'] - sales['return_qty']
sales['qty'] = np.log1p(sales['qty'])
sales.drop(['item_qty', 'return_qty'], axis=1, inplace=True)

# 每个销售中心每个客户每个产品每天的销量
sales = sales.unstack(level=-1).fillna(0)
sales.columns = sales.columns.get_level_values(1)

items = items.reindex(sales.index.get_level_values(2))
centers = centers.reindex(sales.index.get_level_values(0))

# 每个产品每天的销量
item_sales = sales.groupby(['item_code'])[sales.columns].sum()

# 每个客户每个产品每天的销量
customer_item_sales = sales.reset_index()
customer_item_sales_index = customer_item_sales[['customer_wid', 'item_code']]
customer_item_sales = sales.groupby(['customer_wid', 'item_code'])[sales.columns].sum()

# 每个销售中心每个产品每天的销量
center_item_sales = sales.reset_index()
center_item_sales_index = center_item_sales[['sales_cen_wid', 'item_code']]
center_item_sales = sales.groupby(['sales_cen_wid', 'item_code'])[sales.columns].sum()


# ============================================================================================== #

# ============================================================================================== #
# Part 2: Extract features


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_dataset(df, dt, is_train=True, name_prefix=None):
    X = {}

    # 销量统计特征
    for i in [3, 7, 14, 30, 60, 90]:
        tmp = get_timespan(df, dt, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    # 销量统计特征2
    for i in [3, 7, 14, 30, 60, 90]:
        tmp = get_timespan(df, dt + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    # 有销量的天数特征
    for i in [7, 14, 30, 60, 90]:
        tmp = get_timespan(df, dt, i, i)
        X['has_sales_days_in_prev_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_prev_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_prev_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    # 前15天的销量
    for i in range(1, 16):
        X['prev_%s' % i] = get_timespan(df, dt, i, 1).values.ravel()

    # 前4（12）周每个星期几（星期一到星期日）的平均销量
    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df, dt, 28 - i, 4, freq='7D').mean(axis=1).values
        X['mean_12_dow{}'.format(i)] = get_timespan(df, dt, 84 - i, 12, freq='7D').mean(axis=1).values

    X = pd.DataFrame(X)

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    if is_train:
        y = df[pd.date_range(dt, periods=30)].values
        return X, y

    return X


# 准备训练集
print("[INFO] Preparing training data...")
t0 = time()

dt = date(2018, 5, 27)
num_days = 6
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(sales, dt + delta)

    X_tmp2 = prepare_dataset(item_sales, dt + delta, is_train=False, name_prefix='item')
    X_tmp2.index = item_sales.index
    X_tmp2 = X_tmp2.reindex(sales.index.get_level_values(2)).reset_index(drop=True)

    X_tmp3 = prepare_dataset(customer_item_sales, dt + delta, is_train=False, name_prefix='customer_item')
    X_tmp3.index = customer_item_sales.index
    X_tmp3 = X_tmp3.reindex(customer_item_sales_index).reset_index(drop=True)

    X_tmp4 = prepare_dataset(center_item_sales, dt + delta, is_train=False, name_prefix='center_item')
    X_tmp4.index = center_item_sales.index
    X_tmp4 = X_tmp4.reindex(center_item_sales_index).reset_index(drop=True)

    # X_tmp5 = prepare_dataset(center_customer_sales, dt + delta, is_train=False, name_prefix='center_customer')
    # X_tmp5 = center_customer_sales.index
    # X_tmp5 = X_tmp5.reindex(center_customer_sales_index).reset_index(drop=False)
    #
    # X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, X_tmp4, X_tmp5, items.reset_index(), centers.reset_index()], axis=1)
    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, X_tmp4, items.reset_index(), centers.reset_index()], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp, y_tmp, X_tmp2, X_tmp3, X_tmp4
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

# 准备验证集
print("[INFO] Preparing validation data...")
t0 = time()

dt = date(2018, 8, 1)
X_val, y_val = prepare_dataset(sales, dt)

X_val2 = prepare_dataset(item_sales, dt, is_train=False, name_prefix='item')
X_val2.index = item_sales.index
X_val2 = X_val2.reindex(sales.index.get_level_values(2)).reset_index(drop=True)

X_val3 = prepare_dataset(customer_item_sales, dt, is_train=False, name_prefix='customer_item')
X_val3.index = customer_item_sales.index
X_val3 = X_val3.reindex(customer_item_sales_index).reset_index(drop=True)

X_val4 = prepare_dataset(center_item_sales, dt, is_train=False, name_prefix='customer_item')
X_val4.index = center_item_sales.index
X_val4 = X_val4.reindex(center_item_sales_index).reset_index(drop=True)

# X_val5 = prepare_dataset(center_customer_sales, dt, is_train=False, name_prefix='center_customer')
# X_val5.index = center_customer_sales.index
# X_val5 = X_val5.reindex(center_customer_sales_index).reset_index(drop=False)
#
# X_val = pd.concat([X_val, X_val2, X_val3, X_val4, X_val5, items.reset_index(), centers.reset_index()], axis=1)
X_val = pd.concat([X_val, X_val2, X_val3, X_val4, items.reset_index(), centers.reset_index()], axis=1)

del X_val2, X_val3, X_val4
gc.collect()

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds" % (time() - t0))

# ============================================================================================== #

# ============================================================================================== #
# Part 3: Train a model and predict

print("[INFO] Start training and predicting...")
t0 = time()

params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}

MAX_ROUNDS = 5000
val_pred = []
test_pred = []
cate_vars = []
for i in range(30):
    print('=' * 50)
    print("Step %d" % (i + 1))
    print('=' * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        categorical_feature=cate_vars
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125,
        verbose_eval=50
    )
    print('\n'.join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance('gain')), key=lambda x: x[1], reverse=True)))
    val_pred.append(
        bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS)
    )

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

print("Validation mse:", mean_squared_error(y_val, np.array(val_pred).transpose()))

err = (y_val - np.array(val_pred).transpose()) ** 2
err = err.sum(axis=1)
err = np.sqrt(err.sum() / 30)
print("nwrsle = {}".format(err))

df_val = pd.DataFrame(
    y_val, index=sales.index,
    columns=pd.date_range('2018-08-01', periods=30)
).stack().to_frame('qty')
df_val.index.set_names(['sales_cen_wid', 'customer_wid', 'item_code'], inplace=True)
df_val['qty'] = np.expm1(df_val['qty'])
df_val.reset_index().to_csv('val.csv', index=False)

df_preds = pd.DataFrame(
    np.array(val_pred).transpose(), index=sales.index,
    columns=pd.date_range('2018-08-01', periods=30)
).stack().to_frame('qty')
df_preds.index.set_names(['sales_cen_wid', 'customer_wid', 'item_code'], inplace=True)
df_preds['qty'] = np.expm1(df_preds['qty'])
df_preds.reset_index().to_csv('lgb_cv.csv', index=False)

# ============================================================================================== #
