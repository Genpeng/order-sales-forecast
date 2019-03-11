# _*_ coding: utf-8 _*_

"""
Use AdaBoost model to predict demand of order, where the base estimator is Decision Tree.

Author: Genpeng Xu
Date:   2019/03/04
"""

# Import necessary libraries
import gc
import numpy as np
import pandas as pd
from time import time
from datetime import date
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ============================================================================================= #
# 1 数据预处理

# 载入数据
order = pd.read_csv("../data/level2/m111-sku-order.csv", sep=',', parse_dates=['order_date'])
dis = pd.read_csv("../data/level2/m111-sku-dis.csv", sep=',', parse_dates=['dis_date'])
inv = pd.read_csv(
    "../data/level2/m111-sku-inv.csv", sep=',', parse_dates=['period_wid']
).rename(columns={'period_wid': 'inv_date'})
category = pd.read_csv(
    "../data/level2/item2category-from-3.csv", sep=','
).rename(columns={'sales_segment1_code': 'category'})

# 考虑的品类有：消毒柜、洗碗机、烟机、灶具、电热、净水机、燃热、饮水机
cates_considered = ['CRXDG', 'CRXWJ', 'CRYJ', 'CRZJ', 'DR', 'JSJ', 'RR', 'YSJ']

# 处理提货数据
order = order.loc[(order.order_date >= '2017-01-01') & (order.order_date <= '2018-12-31')]
order = order.join(category.set_index('item_code'), on='item_code', how='left')
order = order.loc[order.category.isin(cates_considered)]
df_test = order.loc[order.order_date >= '2018-11-01']  # 测试集
order = order.loc[order.order_date <= '2018-10-31']  # 训练和验证集
df_test['month'] = df_test.order_date.astype('str').apply(lambda x: x[:7])
df_test = df_test.groupby(['category', 'month'])[['qty']].sum()
order_cate_month = order.copy()
order_cate_month['month'] = order_cate_month.order_date.astype('str').apply(lambda x: x[:7])
order_cate_month = order_cate_month.groupby(['category', 'month'])[['qty']].sum()
# order_cate_month['qty'] = np.log1p(order_cate_month.qty)
order_cate_month = order_cate_month.unstack(level=-1).fillna(0)
order_cate_month.columns = pd.date_range('2017-01-31', '2018-10-31', freq='M')

# 处理分销数据
dis = dis.loc[(dis.dis_date >= '2017-06-01') & (dis.dis_date <= '2018-10-31')]
dis = dis.join(category.set_index('item_code'), on='item_code', how='left')
dis = dis.loc[dis.category.isin(cates_considered)]
dis_cate_month = dis.copy()
dis_cate_month['month'] = dis_cate_month.dis_date.astype('str').apply(lambda x: x[:7])
dis_cate_month = dis_cate_month.groupby(['category', 'month'])[['qty']].sum()
# dis_cate_month['qty'] = np.log1p(dis_cate_month.qty)
dis_cate_month = dis_cate_month.unstack(level=-1).fillna(0)
dis_cate_month.columns = pd.date_range('2017-06-30', '2018-10-31', freq='M')
dis_cate_month = dis_cate_month.reindex(order_cate_month.index).fillna(0)

# 处理库存数据
inv = inv.loc[(inv.inv_date >= '2017-12-01') & (inv.inv_date <= '2018-10-31')]
inv = inv.join(category.set_index('item_code'), on='item_code', how='left')
inv = inv.loc[inv.category.isin(cates_considered)]
inv_lastday = inv.loc[inv.inv_date.isin(pd.date_range('2017-12-31', '2018-10-31', freq='M'))]
inv_cate_month = inv_lastday.copy()
inv_cate_month['month'] = inv_cate_month.inv_date.astype('str').apply(lambda x: x[:7])
inv_cate_month = inv_cate_month.groupby(['category', 'month'])[['qty']].sum()
# inv_cate_month['qty'] = np.log1p(inv_cate_month.qty)
inv_cate_month = inv_cate_month.unstack(level=-1).fillna(0)
inv_cate_month.columns = pd.date_range('2017-12-31', '2018-10-31', freq='M')
inv_cate_month = inv_cate_month.reindex(order_cate_month.index).fillna(0)

# 处理品类特征
cates = pd.DataFrame(cates_considered, columns=['category'])
encoder = LabelEncoder()
cates['cate_enc'] = encoder.fit_transform(cates.category)
cates = cates.set_index('category').reindex(order_cate_month.index)

# ============================================================================================= #

# ============================================================================================= #
# 2 特征工程


def prepare_dataset(order, dis, inv, year, month, is_train=True, name_prefix=None):
    X = {}

    # 提货的统计特征
    for i in [3]:
        dt = date(year, month, 1)
        tmp = order[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月提货量
        X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
        X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月提货量的和（带衰减）
        X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
        X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
        X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
        X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
        X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    #     for i in [3]:
    #         dt = date(year, month, 1)
    #         tmp = dis[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月分销量
    #         X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
    #         X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月分销量的和（带衰减）
    #         X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
    #         X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
    #         X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
    #         X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
    #         X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    #     for i in [3, 6]:
    #         dt = date(year, month, 1)
    #         tmp = inv[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月库存量
    #         X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
    #         X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月库存量的和（带衰减）
    #         X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
    #         X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
    #         X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
    #         X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
    #         X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 前3个月的提货量
    for i in range(1, 4):
        if month - i <= 0:
            start_dt = date(year - 1, month + 12 - i, 1)
        else:
            start_dt = date(year, month - i, 1)
        X['ord_pre_%s' % i] = order[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前3个月分销量
    #     for i in range(1, 4):
    #         if month - i <= 0:
    #             start_dt = date(year - 1, month + 12 - i, 1)
    #         else:
    #             start_dt = date(year, month - i, 1)
    #         X['dis_pre_%s' % i] = dis[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    # 前6个月的库存量
    #     for i in range(1, 7):
    #         if month - i <= 0:
    #             start_dt = date(year - 1, month + 12 - i, 1)
    #         else:
    #             start_dt = date(year, month - i, 1)
    #         X['inv_pre_%s' % i] = inv[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()

    X = pd.DataFrame(X)

    if is_train:
        start_dt = date(year, month, 1)
        y = order[pd.date_range(start_dt, periods=2, freq='M')].values
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X


def get_pre_10_days(order, dis, inv, index, year, month):
    X = {}
    start_dt, end_dt = date(year, month, 1), date(year, month, 10)

    # 每个品类M月前10天的提货量
    ord_tmp = order.loc[order.order_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
    ord_tmp = ord_tmp.groupby('category')[['qty']].sum()
    ord_tmp = ord_tmp.reindex(index).fillna(0)
    #     ord_tmp['qty'] = ord_tmp.qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
    X['ord_pre_10_days'] = ord_tmp.values.ravel()

    # 每个品类M月前10天的分销量
    #     dis_tmp = dis.loc[dis.dis_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
    #     dis_tmp = dis_tmp.groupby('category')[['qty']].sum()
    #     dis_tmp = dis_tmp.reindex(index).fillna(0)
    #     dis_tmp['qty'] = dis_tmp.qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
    #     X['dis_pre_10_days'] = dis_tmp.values.ravel()

    # 每个品类M月前10天的库存
    #     inv_tmp = inv.loc[inv.inv_date.isin(pd.date_range(end=end_dt, periods=1, freq='D'))]
    #     inv_tmp = inv_tmp.groupby('category')[['qty']].sum()
    #     inv_tmp = inv_tmp.reindex(index).fillna(0)
    #     inv_tmp['qty'] = inv_tmp.qty.apply(lambda x: np.log1p(x) if x > 0 else 0)
    #     X['inv_pre_10_days'] = inv_tmp.values.ravel()

    X = pd.DataFrame(X)

    return X


# 准备训练数据
train_month = [
    '2017-05',
    '2017-06',
    '2017-07',
    '2017-08',
    '2017-09',
    '2017-10',
    '2017-11',
    '2017-12',
    '2018-01',
    '2018-02',
    '2018-03',
    '2018-04',
    '2018-05',
    '2018-06',
    '2018-07'
]
X_l, y_l = [], []
for month in train_month:
    y, m = int(month.split('-')[0]), int(month.split('-')[1])
    pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, y, m)
    X_tmp, y_tmp = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, y, m)
    X_tmp = pd.concat([X_tmp, cates.reset_index(drop=True)], axis=1)
    #     X_tmp = pd.concat([X_tmp, pre_10_days, cates.reset_index(drop=True)], axis=1)
    X_tmp['pred_month'] = m
    X_tmp['is_spring_fest'] = X_tmp.pred_month.apply(
        lambda x: 1 if (y == 2017 and x == 1) or (y == 2018 and x == 2) else 0)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp, y_tmp
    gc.collect()
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

# 准备验证集
pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, 2018, 9)
X_val, y_val = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, 2018, 9)
X_val = pd.concat([X_val, cates.reset_index(drop=True)], axis=1)
# X_val = pd.concat([X_val, pre_10_days, cates.reset_index(drop=True)], axis=1)
X_val['pred_month'] = 9
X_val['is_spring_fest'] = X_val.pred_month.apply(lambda x: 1 if (y == 2017 and x == 1) or (y == 2018 and x == 2) else 0)

# 准备测试集
pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, 2018, 11)
X_test = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, 2018, 11, is_train=False)
X_test = pd.concat([X_test, cates.reset_index(drop=True)], axis=1)
# X_test = pd.concat([X_test, pre_10_days, cates.reset_index(drop=True)], axis=1)
X_test['pred_month'] = 11
X_test['is_spring_fest'] = X_test.pred_month.apply(lambda x: 1 if (y == 2017 and x == 1) or (y == 2018 and x == 2) else 0)

# ============================================================================================= #

# ============================================================================================= #
# 3 训练和预测


def error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def acc(y_true, y_pred):
    return 1 - error(y_true, y_pred)


print("[INFO] Start training and predicting...")
t0 = time()

regr = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=8),
    n_estimators=1200, random_state=np.random.RandomState(1)
)

preds_train, preds_val, preds_test = [], [], []
accs_train, accs_val = [], []
mses_train, mses_val = [], []

for i in range(2):
    print()
    print('# ' + '=' * 100 + ' #')
    print('# ' + 'Step %d' % i + ' ' * (100 - len('Step %d' % i)) + ' #')
    print('# ' + '=' * 100 + ' #')

    # Add previous predictions as a new feature
    if preds_train:
        X_train['m%s' % (i - 1)] = pd.Series(preds_train[i - 1])
        X_val['m%s' % (i - 1)] = pd.Series(preds_val[i - 1])
        X_test['m%s' % (i - 1)] = pd.Series(preds_test[i - 1])

    # Adjust the month predicted
    if i != 0:
        X_train['pred_month'] = X_train.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)
        X_val['pred_month'] = X_val.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)
        X_test['pred_month'] = X_test.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)

    print("[INFO] Fit the model...")
    regr.fit(X_train.values, y_train[:, i])

    # Predict
    pred_train = regr.predict(X_train.values)
    pred_val = regr.predict(X_val.values)
    pred_test = regr.predict(X_test.values)

    # Calculate accuracy
    acc_train = acc(y_train[:, i], pred_train)
    acc_val = acc(y_val[:, i], pred_val)
    print("[INFO] The accuracy of training set is: %.2f%%\t The accuracy of validation set is: %.2f%%" % (acc_train * 100, acc_val * 100))

    # Calculate MSE
    mse_train = mean_squared_error(y_train[:, i], pred_train)
    mse_val = mean_squared_error(y_val[:, i], pred_val)
    print("[INFO] The MSE of training set is: %.2f%%\t The MSE of validation set is: %.2f%%" % (acc_train * 100, acc_val * 100))

    # Store the intermediate results
    preds_train.append(pred_train)
    preds_val.append(pred_val)
    preds_test.append(pred_test)
    accs_train.append(acc_train)
    accs_val.append(acc_val)
    mses_train.append(mse_train)
    mses_val.append(mse_val)

print()
print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

print()
print("The MSE error of validation set is:", mean_squared_error(y_val, np.array(preds_val).transpose()))

# ============================================================================================= #

# ============================================================================================= #
# 4 结果评估

# 归一化加权均方根对数误差
err = (y_val - np.array(preds_val).transpose()) ** 2
err = err.sum(axis=1)
err = np.sqrt(err.sum() / 2 / len(y_val))
print("The NWRMSLE error of validation set is:", err)

# 验证集（9-10月）准确率（品类）
df_val = pd.DataFrame(
    np.array(y_val), index=order_cate_month.index,
    columns=['2018-09', '2018-10']
)
df_pred_val = pd.DataFrame(
    np.array(preds_val).transpose(), index=order_cate_month.index,
    columns=['2018-09', '2018-10']
)
m_acc = acc(df_val['2018-09'], df_pred_val['2018-09'])
m1_acc = acc(df_val['2018-10'], df_pred_val['2018-10'])
print("The accuracy of 'M' order amount is: %.2f%%" % (m_acc * 100))
print("The accuracy of 'M+1' order amount is: %.2f%%" % (m1_acc * 100))

# 测试集（11-12月）准确率（品类）
df_pred_test = pd.DataFrame(
    np.array(preds_test).transpose(), index=order_cate_month.index,
    columns=['2018-11', '2018-12']
).stack().to_frame('pred_qty')
df_pred_test.index.set_names(['category', 'month'], inplace=True)
comp = df_test.join(df_pred_test, how='left').fillna(0).reset_index()
m_comp = comp.loc[comp['month'] == '2018-11']
m1_comp = comp.loc[comp['month'] == '2018-12']
m_acc = acc(m_comp['qty'], m_comp['pred_qty'])
m1_acc = acc(m1_comp['qty'], m1_comp['pred_qty'])
print("The accuracy of 'M' order amount is: %.2f%%" % (m_acc * 100))
print("The accuracy of 'M+1' order amount is: %.2f%%" % (m1_acc * 100))

# ============================================================================================= #
