# _*_ coding: utf-8 _*_

import gc
import os
import time
import json
import requests
import calendar
import numpy as np
import pandas as pd
import lightgbm as lgb
from math import sqrt
from datetime import date
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Own customized modules
from core.global_variables import (ORDER_DATA_DIR,
                                   ORDER_DATA_COLUMN_NAMES,
                                   ALL_CATE_CODES)
from core.util.data_util import (transform_channel,
                                 transform_project_flag,
                                 remove_whitespace)
from core.util.date_util import get_curr_date, infer_month

# curr_year, curr_month, _ = get_curr_date()
curr_year, curr_month, _ = 2019, 11, 16
start_pred_year, start_pred_month, start_pred_day = 2019, 12, 10
periods = 3
# data_file_flag = "%d-%02d" % (curr_year, curr_month)

# 1 数据预处理
## 1.1 提货数据


def read_and_preprocess_order_data(year, month, need_unitize=True):
    data_file_flag = "%d-%02d" % (year, month)
    ord_path = os.path.join(ORDER_DATA_DIR, '%s/m111-order_%s.txt' % tuple([data_file_flag] * 2))
    order = pd.read_csv(ord_path, sep='\u001e', header=None, names=ORDER_DATA_COLUMN_NAMES, parse_dates=[0])
    removed_column_names = ['org_code', 'region_code', 'region_name', 'road',
                            'fin_segment1_code', 'fin_segment1_name',
                            'fin_segment2_code', 'fin_segment2_name',
                            'received_qty', 'return_qty']
    order.drop(columns=removed_column_names, inplace=True)
    order = order.loc[order.first_cate_code.isin(ALL_CATE_CODES)]
    order = order.loc[order.order_date >= '2015-09-01']
    order = order.sort_values(by='order_date').reset_index(drop=True)
    str_column_names = ['sales_cen_code', 'customer_code', 'item_code']
    remove_whitespace(order, str_column_names)
    order['district'] = order.district.str.replace(r'\\N', '未知')
    order['channel_name'] = order.channel_name.apply(lambda x: transform_channel(x))
    order['channel_name'] = order.channel_name.str.replace(r'\\N', '未知')
    order['sales_chan_name'] = order.sales_chan_name.str.replace(r'\\N', '未知')
    order.project_flag.fillna('未知', inplace=True)
    order['project_flag'] = order.project_flag.apply(lambda x: transform_project_flag(x))
    if need_unitize:
        order['ord_qty'] = order.ord_qty / 10000
        order['ord_amount'] = order.ord_amount / 10000
    return order


# In[12]:


def get_days_of_month(year, month):
    return calendar.monthrange(year, month)[1]


# In[13]:


# 每个SKU每个月的提货量
order_sku_month = order.copy()
order_sku_month['order_month'] = order_sku_month.order_date.astype(str).apply(lambda x: x[:7])
order_sku_month = order_sku_month.groupby(['item_code', 'order_month'])[['ord_qty']].sum()
order_sku_month['ord_qty'] = order_sku_month.ord_qty.apply(lambda x: 0 if x < 0 else x)
order_sku_month = order_sku_month.unstack(level=-1).fillna(0.0)
end_dt_str = '%d-%02d-%d' % (curr_year,
                             curr_month,
                             get_days_of_month(curr_year, curr_month))
order_sku_month.columns = pd.date_range(start='2015-09-30', end=end_dt_str, freq='M')
# order_sku_month.columns = pd.date_range(start='2018-01-31', end='2019-07-31', freq='M')


# ## 1.2 分销数据

# In[14]:


# curr_year, curr_month, _ = 2019, 10, 10
# data_file_flag = "%d-%02d" % (curr_year, curr_month)
dis_path = "/data/aidev/order-sales-forecast_old/data/dis/%s/m111-dis_%s.txt" % tuple([data_file_flag] * 2)
column_names = ['order_date',
                'bu_code',
                'bu_name',
                'customer_code',
                'customer_name',
                'sales_cen_code',
                'sales_cen_name',
                'sales_region_code',
                'sales_region_name',
                'region_code',
                'region_name',
                'province',
                'city',
                'district',
                'road',
                'item_code',
                'item_name',
                'first_cate_code',
                'first_cate_name',
                'second_cate_code',
                'second_cate_name',
                'customer_type',
                'is_usable',
                'channel_name',
                'channel_level',
                'sales_chan_name',
                'dis_qty',
                'item_price',
                'dis_amount']
dis = pd.read_csv(dis_path, sep='\u001e', header=None, names=column_names, parse_dates=[0])
dis.drop(columns=['bu_code',
                  'bu_name',
                  'region_code',
                  'region_name',
                  'road'], inplace=True)
dis = dis.loc[dis.first_cate_code.isin(ALL_CATE_CODES)]
dis = dis.loc[dis.customer_code.str.startswith('C')]
dis = dis.sort_values(by='order_date').reset_index(drop=True)
dis['sales_cen_code'] = dis.sales_cen_code.astype(str).apply(lambda x: x.strip())
dis['customer_code'] = dis.customer_code.astype(str).apply(lambda x: x.strip())
dis['item_code'] = dis.item_code.astype(str).apply(lambda x: x.strip())
dis['district'] = dis.district.str.replace(r'\\N', '未知')
dis['customer_type'] = dis.customer_type.str.replace(r'\\N', '未知')
dis['is_usable'] = dis.is_usable.astype(str)
dis['is_usable'] = dis.is_usable.str.replace(r'\\N', '未知')
dis['channel_name'] = dis.channel_name.apply(lambda x: transform_channel(x))
dis['dis_qty'] = np.round(dis.dis_qty).astype(int)
dis['item_price'] = dis.item_price.astype(str).str.replace(r'\\N', '-1.0').astype(float)
dis['dis_amount'] = dis.dis_qty * dis.item_price
dis['dis_qty'] = dis.dis_qty / 10000
dis['dis_amount'] = dis.dis_amount / 10000

# In[15]:


# 每个SKU每个月的分销量
dis_sku_month = dis.copy()
dis_sku_month['order_month'] = dis_sku_month.order_date.astype(str).apply(lambda x: x[:7])
dis_sku_month = dis_sku_month.groupby(['item_code', 'order_month'])[['dis_qty']].sum()
dis_sku_month['dis_qty'] = dis_sku_month.dis_qty.apply(lambda x: 0 if x < 0 else x)
dis_sku_month = dis_sku_month.unstack(level=-1).fillna(0.0)
end_dt_str = '%d-%02d-%d' % (curr_year,
                             curr_month,
                             get_days_of_month(curr_year, curr_month))
# dis_sku_month.columns = pd.date_range(start='2015-09-30', end=end_dt_str, freq='M')
dis_sku_month.columns = pd.date_range(start='2018-01-31', end=end_dt_str, freq='M')
# dis_sku_month.columns = pd.date_range(start='2018-01-31', end='2019-09-30', freq='M')


# ## 1.3 库存数据

# In[16]:


# curr_year, curr_month, _ = 2019, 10, 10
# data_file_flag = "%d-%02d" % (curr_year, curr_month)
inv_path = "/data/aidev/order-sales-forecast_old/data/inv/%s/m111-inv_%s.txt" % tuple([data_file_flag] * 2)
column_names = ['order_date',
                'bu_code',
                'bu_name',
                'sales_cen_code',
                'sales_cen_name',
                'sales_region_code',
                'sales_region_name',
                'region_code',
                'region_name',
                'province',
                'city',
                'district',
                'customer_code',
                'customer_name',
                'customer_type',
                'is_usable',
                'channel_name',
                'channel_level',
                'sales_chan_name',
                'item_code',
                'item_name',
                'first_cate_code',
                'first_cate_name',
                'second_cate_code',
                'second_cate_name',
                'inv_qty',
                'inv_amount',
                'item_price']
inv = pd.read_csv(inv_path, sep='\u001e', header=None, names=column_names, parse_dates=[0])
inv.drop(columns=['bu_code',
                  'bu_name',
                  'region_code',
                  'region_name'], inplace=True)
inv = inv.loc[inv.first_cate_code.isin(ALL_CATE_CODES)]
inv = inv.sort_values(by='order_date').reset_index(drop=True)
inv['sales_cen_code'] = inv.sales_cen_code.astype(str).apply(lambda x: x.strip())
inv['customer_code'] = inv.customer_code.astype(str).apply(lambda x: x.strip())
inv['item_code'] = inv.item_code.astype(str).apply(lambda x: x.strip())
inv['district'] = inv.district.str.replace(r'\\N', '未知')
inv['channel_name'] = inv.channel_name.apply(lambda x: transform_channel(x))
inv['sales_chan_name'] = inv.sales_chan_name.str.replace(r'\\N', '未知')
inv['inv_qty'] = inv.inv_qty / 10000
inv['inv_amount'] = inv.inv_amount / 10000

# In[17]:


# 每个SKU每个月的库存量
inv_sku_month = inv.copy()
inv_sku_month['order_month'] = inv_sku_month.order_date.astype(str).apply(lambda x: x[:7])
inv_sku_month = inv_sku_month.groupby(['item_code', 'order_month'])[['inv_qty']].sum()
inv_sku_month['inv_qty'] = inv_sku_month.inv_qty.apply(lambda x: 0 if x < 0 else x)
inv_sku_month = inv_sku_month.unstack(level=-1).fillna(0.0)
end_dt_str = '%d-%02d-%d' % (curr_year,
                             curr_month,
                             get_days_of_month(curr_year, curr_month))
# inv_sku_month.columns = pd.date_range(start='2015-09-30', end=end_dt_str, freq='M')
inv_sku_month.columns = pd.date_range(start='2017-04-30', end=end_dt_str, freq='M')
# inv_sku_month.columns = pd.date_range(start='2017-04-30', end='2019-09-30', freq='M')


# ## 1.3 SKU信息

# In[18]:


engine = create_engine("mysql+pymysql://readonly:readonly@3306-W-BD-UAT-TLY01-MYC5.service.dcnh.consul:3306/apd")
version_flag = "%d-%02d" % (curr_year, curr_month)
sql = "SELECT * FROM m111_item_list WHERE version = '%s'" % version_flag
items = pd.read_sql_query(sql, engine)
column_names = ['id',
                'item_code',
                'item_name',
                'first_cate_code',
                'state',
                'manu_code',
                'need_calc',
                'is_highend',
                'appear_date',
                'delisting_date',
                'scheduled_delisting_date',
                'six_eighteen_main_product_date',
                'double_eleven_main_product_date',
                'last_update_date',
                'last_updated_by',
                'creation_date',
                'created_by',
                'version']
items.columns = column_names
str_cols = ['item_code', 'item_name', 'first_cate_code', 'state',
            'manu_code', 'need_calc', 'is_highend', 'appear_date',
            'delisting_date', 'scheduled_delisting_date',
            'six_eighteen_main_product_date', 'double_eleven_main_product_date']
remove_whitespace(items, str_cols)
dt_cols = ['appear_date', 'delisting_date',
           'scheduled_delisting_date',
           'six_eighteen_main_product_date',
           'double_eleven_main_product_date']
for col in dt_cols:
    items[col] = items[col].apply(lambda x: x[:7])
    items[col] = pd.to_datetime(items[col])

# In[19]:


# SKU列表
# items = pd.read_csv(
#     "../../data/opt/item_list.txt", sep='\t', 
#     parse_dates=['appear_date', 'delisting_date', 
#                  'scheduled_delisting_date', 
#                  'six_eighteen_main_product_date', 
#                  'double_eleven_main_product_date']
# )
# items['item_code'] = items.item_code.astype(str).apply(lambda x: x.strip())

# items['appear_date'] = items.appear_date.fillna('')
# items['delisting_date'] = items.delisting_date.fillna('')
# items['scheduled_delisting_date'] = items.scheduled_delisting_date.fillna('')
# items['six_eighteen_main_product_date'] = items.six_eighteen_main_product_date.fillna('')
# items['double_eleven_main_product_date'] = items.double_eleven_main_product_date.fillna('')


# In[20]:


# 需要计算的SKU
# white_list_path = '../../data/opt/white-list-%d%02d.txt' % (true_pred_year, true_pred_month)
# white_list = pd.read_csv(white_list_path, sep='\t')
# white_list['item_code'] = white_list.item_code.astype(str).apply(lambda x: x.strip())
# white_items = set(white_list.item_code.unique())

white_items = set(items.loc[items.need_calc == '是'].item_code.unique())

# In[21]:


# 规划退市的SKU
scheduled_delisting_items = items.loc[~items.scheduled_delisting_date.isna()]
start_dt = '%d-%02d-01' % infer_month(true_pred_year, true_pred_month, -2)
months_need_considered = pd.date_range(start=start_dt, periods=3, freq=pd.offsets.MonthBegin(1))
scheduled_delisting_items = set(scheduled_delisting_items.loc[scheduled_delisting_items.scheduled_delisting_date.isin(
    months_need_considered)].item_code.unique())

# In[22]:


# 退市的SKU
delisting_items = items.loc[~items.delisting_date.isna()]
end_dt = '%d-%02d-01' % (start_pred_year, start_pred_month)
months_need_considered = pd.date_range(start='2015-01-01', end=end_dt, freq=pd.offsets.MonthBegin(1))
delisting_items = set(
    delisting_items.loc[delisting_items.delisting_date.isin(months_need_considered)].item_code.unique())

# In[23]:


# 新品的SKU
new_items = items.loc[~items.appear_date.isna()]
end_dt = '%d-%02d-01' % (true_pred_year, true_pred_month)
months_need_considered = pd.date_range(end=end_dt, periods=3, freq=pd.offsets.MonthBegin(1))
new_items = set(new_items.loc[new_items.appear_date.isin(months_need_considered)].item_code.unique())

# In[24]:


# 当月新品
curr_new_items = items.loc[~items.appear_date.isna()]
end_dt = '%d-%02d-01' % (true_pred_year, true_pred_month)
months_need_considered = pd.date_range(end=end_dt, periods=1, freq=pd.offsets.MonthBegin(1))
curr_new_items = set(curr_new_items.loc[curr_new_items.appear_date.isin(months_need_considered)].item_code.unique())

# In[25]:


# 618主推的SKU
six_eighteen_main_items = items.loc[~items.six_eighteen_main_product_date.isna()]
end_dt = '%d-%02d-01' % (true_pred_year, 6)
months_need_considered = pd.date_range(end=end_dt, periods=3, freq=pd.offsets.MonthBegin(1))
six_eighteen_main_items = set(six_eighteen_main_items.loc[six_eighteen_main_items.six_eighteen_main_product_date.isin(
    months_need_considered)].item_code.unique())

# In[26]:


# 双11主推的SKU
double_eleven_main_items = items.loc[~items.double_eleven_main_product_date.isna()]
end_dt = '%d-%02d-01' % (true_pred_year, 11)
months_need_considered = pd.date_range(end=end_dt, periods=3, freq=pd.offsets.MonthBegin(1))
double_eleven_main_items = set(double_eleven_main_items.loc[
                                   double_eleven_main_items.six_eighteen_main_product_date.isin(
                                       months_need_considered)].item_code.unique())

# In[27]:


# SKU信息
sku_info = order.drop_duplicates(['item_code'], keep='last')
sku_info = sku_info[['item_code',
                     'item_name',
                     'first_cate_code',
                     'first_cate_name',
                     'second_cate_code',
                     'second_cate_name',
                     'item_price',
                     'channel_name',
                     'sales_chan_name',
                     'project_flag']]
sku_info['need_calc'] = sku_info.item_code.apply(lambda x: 1 if x in white_items else 0)
# sku_info['is_new'] = sku_info.item_code.apply(lambda x: 1 if x in new_items else 0)
# sku_info['is_curr_new'] = sku_info.item_code.apply(lambda x: 1 if x in curr_new_items else 0)
sku_info['is_delisting'] = sku_info.item_code.apply(lambda x: 1 if x in delisting_items else 0)
sku_info['is_scheduled_delisting'] = sku_info.item_code.apply(lambda x: 1 if x in scheduled_delisting_items else 0)
sku_info['is_618_main_products'] = sku_info.item_code.apply(lambda x: 1 if x in six_eighteen_main_items else 0)
sku_info['is_1111_main_products'] = sku_info.item_code.apply(lambda x: 1 if x in double_eleven_main_items else 0)
temp = items.set_index('item_code').to_dict()
sku_info['state'] = sku_info.item_code.map(temp['state']).fillna('其他')
sku_info['is_highend'] = sku_info.item_code.map(temp['is_highend']).fillna('其他')

del temp
gc.collect()

# Ordinal Encoding
label_enc = LabelEncoder()
sku_info['first_cate_id'] = label_enc.fit_transform(sku_info.first_cate_code)
sku_info['second_cate_id'] = label_enc.fit_transform(sku_info.second_cate_code)
sku_info['state'] = label_enc.fit_transform(sku_info.state)
sku_info['is_highend'] = label_enc.fit_transform(sku_info.is_highend)
sku_info['channel_id'] = label_enc.fit_transform(sku_info.channel_name)
sku_info['sales_chan_id'] = label_enc.fit_transform(sku_info.sales_chan_name)
sku_info['project_flag_id'] = label_enc.fit_transform(sku_info.project_flag)
sku_info_filtered = sku_info.drop(
    columns=['item_name', 'first_cate_code', 'first_cate_name',
             'second_cate_code', 'second_cate_name', 'channel_name',
             'sales_chan_name', 'project_flag']
).set_index('item_code')
sku_info = sku_info.set_index('item_code').reindex(order_sku_month.index)
sku_info_filtered = sku_info_filtered.reindex(order_sku_month.index)

# Onehot Encoding
# onehot_column_names = ['first_cate_code', 'second_cate_code', 'state', 'is_highend',
#                        'channel_name', 'sales_chan_name', 'project_flag']
# for cn in onehot_column_names:
#     enc = OneHotEncoder()
#     enc_vals = enc.fit_transform(sku_info[cn])
#     sku_info = pd.concat([sku_info, enc_vals], axis=1)
# label_enc = LabelEncoder()
# sku_info['first_cate_id'] = label_enc.fit_transform(sku_info.first_cate_code)
# sku_info['second_cate_id'] = label_enc.fit_transform(sku_info.second_cate_code)
# sku_info_filtered = sku_info.drop(
#     columns=['item_name', 'first_cate_code', 'first_cate_name',
#              'second_cate_code', 'second_cate_name', 'first_cate_id',
#              'second_cate_id', 'state', 'is_highend', 'channel_name',
#              'sales_chan_name', 'project_flag']
# ).set_index('item_code')
# sku_info = sku_info.set_index('item_code').reindex(order_sku_month.index)
# sku_info_filtered = sku_info_filtered.reindex(order_sku_month.index)


# ## 1.4 衍生数据

# In[28]:


# 每个大类每个月的提货量
order_cate1_month = order_sku_month.reset_index()
order_cate1_month['first_cate_id'] = sku_info.first_cate_id.values
order_cate1_month_index = order_cate1_month['first_cate_id']
order_cate1_month = order_cate1_month.groupby(['first_cate_id'])[order_sku_month.columns].sum()
order_cate1_month = order_cate1_month.reindex(order_cate1_month_index)

# In[29]:


# 每个小类每个月的提货量
order_cate2_month = order_sku_month.reset_index()
order_cate2_month['second_cate_id'] = sku_info.second_cate_id.values
order_cate2_month_index = order_cate2_month['second_cate_id']
order_cate2_month = order_cate2_month.groupby(['second_cate_id'])[order_sku_month.columns].sum()
order_cate2_month = order_cate2_month.reindex(order_cate2_month_index)

# In[30]:


# 每个SKU每个月前15天的提货量（乘以2）
tmp = order.copy()
tmp['day'] = tmp.order_date.dt.day
tmp = tmp.loc[tmp.day <= 15]
tmp['order_month'] = tmp.order_date.astype(str).apply(lambda x: x[:7])
order_sku_pre_15 = tmp.groupby(['item_code', 'order_month'])[['ord_qty']].sum()
order_sku_pre_15['ord_qty'] = order_sku_pre_15.ord_qty.apply(lambda x: 0 if x < 0 else x)
order_sku_pre_15 = order_sku_pre_15.unstack(level=-1).fillna(0.0)
end_dt_str = '%d-%02d-%d' % (curr_year,
                             curr_month,
                             get_days_of_month(curr_year, curr_month))
order_sku_pre_15.columns = pd.date_range(start='2015-09-30', end=end_dt_str, freq='M')
# order_sku_pre_15.columns = pd.date_range(start='2018-01-31', end='2019-07-31', freq='M')
order_sku_pre_15 = order_sku_pre_15.reindex(order_sku_month.index).fillna(0)
order_sku_pre_15 = order_sku_pre_15 * 2

del tmp
gc.collect()


# # 2 特征工程

# In[31]:


def is_spring_festival_pre(y, m):
    if (y == 2019 and m == 1) or (y == 2018 and m == 1) or (y == 2016 and m == 12) or (y == 2016 and m == 1):
        return 1
    else:
        return 0


def is_spring_festival(y, m):
    if (y == 2019 and m == 2) or (y == 2018 and m == 2) or (y == 2017 and m == 1) or (y == 2016 and m == 2):
        return 1
    else:
        return 0


def is_spring_festival_after(y, m):
    if (y == 2019 and m == 3) or (y == 2018 and m == 3) or (y == 2017 and m == 2) or (y == 2016 and m == 3):
        return 1
    else:
        return 0


# In[32]:


def is_home_decoration(m):
    return 1 if m == 3 else 0


# In[33]:


def is_merger_time(y, m):
    if (y == 2018 and m == 6) or (y == 2018 and m == 7):
        return 1
    else:
        return 0


# In[34]:


def is_six_eighteen_pre_2(m):
    return 1 if m == 4 else 0


def is_six_eighteen_pre_1(m):
    return 1 if m == 5 else 0


def is_six_eighteen(m):
    return 1 if m == 6 else 0


def is_six_eighteen_after(m):
    return 1 if m == 7 else 0


# In[35]:


def is_double_eleven_pre_2(m):
    return 1 if m == 9 else 0


def is_double_eleven_pre_1(m):
    return 1 if m == 10 else 0


def is_double_eleven(m):
    return 1 if m == 11 else 0


# In[36]:


def is_double_twelve(m):
    return 1 if m == 12 else 0


# In[37]:


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


# In[38]:


def prepare_dataset_v2(order, dis, inv, year, month, gap=0, add_cate_feat=True, is_train=True, name_prefix=None):
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
    true_pred_year, true_pred_month = infer_month(year, month, gap)
    X = {}

    # 提货的统计特征
    #     if order is not None:
    #         for i in [3, 6]:
    #             dt = date(year, month, 1)
    #             tmp = order[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月提货量
    #             X['ord_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月提货量的平均一阶差分
    #             X['ord_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月提货量的和（带衰减）
    #             X['ord_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月提货量的平均值
    #             X['ord_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月提货量的中位数
    #             X['ord_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月提货量的最大值
    #             X['ord_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月提货量的最小值
    #             X['ord_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月提货量的标准差

    # 分销的统计特征
    #     if dis is not None:
    #         for i in [3, 6]:
    #             dt = date(year, month, 1)
    #             tmp = dis[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月分销量
    #             X['dis_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月分销量的平均一阶差分
    #             X['dis_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月分销量的和（带衰减）
    #             X['dis_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月分销量的均值
    #             X['dis_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月分销量的中位数
    #             X['dis_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月分销量的最大值
    #             X['dis_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月分销量的最小值
    #             X['dis_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月分销量的标准差

    # 库存的统计特征
    #     if inv is not None:
    #         for i in [3, 6]:
    #             dt = date(year, month, 1)
    #             tmp = inv[pd.date_range(end=dt, periods=i, freq='M')]  # 前i个月库存量
    #             X['inv_diff_mean_pre_%s' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i个月库存量的平均一阶差分
    #             X['inv_sum_decay_pre_%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i个月库存量的和（带衰减）
    #             X['inv_mean_pre_%s' % i] = tmp.mean(axis=1).values  # 前i个月库存量的均值
    #             X['inv_median_pre_%s' % i] = tmp.median(axis=1).values  # 前i个月库存量的中位数
    #             X['inv_max_pre_%s' % i] = tmp.max(axis=1).values  # 前i个月库存量的最大值
    #             X['inv_min_pre_%s' % i] = tmp.min(axis=1).values  # 前i个月库存量的最小值
    #             X['inv_std_pre_%s' % i] = tmp.std(axis=1).values  # 前i个月库存量的标准差

    # 提货天数特征
    if order is not None:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = order[pd.date_range(end=dt, periods=i, freq='M')]
            X['has_ord_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有提货的天数
    #             X['last_ord_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有提货的天数
    #             X['first_ord_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有提货的天数

    # 分销天数特征
    if dis is not None:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = dis[pd.date_range(end=dt, periods=i, freq='M')]
            X['has_dis_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有分销的天数
    #             X['last_dis_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有分销的天数
    #             X['first_dis_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有分销的天数

    # 库存天数特征
    if inv is not None:
        for i in [3, 6]:
            dt = date(year, month, 1)
            tmp = inv[pd.date_range(end=dt, periods=i, freq='M')]
            X['has_inv_pre_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i个月有库存的天数
    #             X['last_inv_pre_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values  # 前i个月距离上一次有库存的天数
    #             X['first_inv_pre_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values  # 前i个月距离第一次有库存的天数

    # 前i个月的提货量
    if order is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            start_dt = date(y_tmp, m_tmp, 1)
            ord_tmp = order[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()
            X['ord_pre_%s' % i] = ord_tmp
    #             X['ord_quantile_pre_%s' % i] = pd.qcut(ord_tmp, q=10000, labels=False, duplicates='drop')

    # 前i个月分销量
    if dis is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            start_dt = date(y_tmp, m_tmp, 1)
            dis_tmp = dis[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()
            X['dis_pre_%s' % i] = dis_tmp
    #             X['dis_quantile_pre_%s' % i] = pd.qcut(dis_tmp, q=10000, labels=False, duplicates='drop')

    # 前i个月的库存量
    if inv is not None:
        for i in range(1, 7):
            y_tmp, m_tmp = infer_month(year, month, offset=-i)
            start_dt = date(y_tmp, m_tmp, 1)
            inv_tmp = inv[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()
            X['inv_pre_%s' % i] = inv_tmp
    #             X['inv_quantile_pre_%s' % i] = pd.qcut(inv_tmp, q=10000, labels=False, duplicates='drop')

    # 历史同期前后3个月的提货
    #     if order is not None:
    #         y_his, m_his = get_month(year, month, offset=-12)  # 历史同期
    #         for i in range(-3, 4):
    #             y_curr, m_curr = get_month(y_his, m_his, offset=i)
    #             start_dt = date(y_curr, m_curr, 1)
    #             X['ord_his_%s' % i] = order[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()

    # 历史同期前后3个月的分销
    #     if dis is not None:
    #         y_his, m_his = get_month(year, month, offset=-12)  # 历史同期
    #         for i in range(-3, 4):
    #             y_curr, m_curr = get_month(y_his, m_his, offset=i)
    #             start_dt = date(y_curr, m_curr, 1)
    #             X['dis_his_%s' % i] = dis[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()

    # 历史同期前后3个月的库存
    #     if inv is not None:
    #         y_his, m_his = get_month(year, month, offset=-12)  # 历史同期
    #         for i in range(-3, 4):
    #             y_curr, m_curr = get_month(y_his, m_his, offset=i)
    #             start_dt = date(y_curr, m_curr, 1)
    #             X['inv_his_%s' % i] = inv[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()

    X = pd.DataFrame(X)

    # Categorical features
    if add_cate_feat:
        X['pred_month'] = true_pred_month
        X['pred_quarter'] = X.pred_month.apply(lambda x: get_quarter(x))

        X['is_spring_fest_pre'] = X.pred_month.apply(lambda x: is_spring_festival_pre(true_pred_year, x))
        X['is_spring_fest'] = X.pred_month.apply(lambda x: is_spring_festival(true_pred_year, x))
        X['is_spring_fest_after'] = X.pred_month.apply(lambda x: is_spring_festival_after(true_pred_year, x))

        #         X['is_home_decoration'] = X.pred_month.apply(lambda x: is_home_decoration(x))

        X['is_merger_time'] = X.pred_month.apply(lambda x: is_merger_time(true_pred_year, x))

        X['is_six_eighteen_pre_2'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_2(x))
        X['is_six_eighteen_pre_1'] = X.pred_month.apply(lambda x: is_six_eighteen_pre_1(x))
        X['is_six_eighteen'] = X.pred_month.apply(lambda x: is_six_eighteen(x))
        X['is_six_eighteen_after'] = X.pred_month.apply(lambda x: is_six_eighteen_after(x))

        X['is_double_eleven_pre_2'] = X.pred_month.apply(lambda x: is_double_eleven_pre_2(x))
        X['is_double_eleven_pre_1'] = X.pred_month.apply(lambda x: is_double_eleven_pre_1(x))
        X['is_double_eleven'] = X.pred_month.apply(lambda x: is_double_eleven(x))

        X['is_double_twelve'] = X.pred_month.apply(lambda x: is_double_twelve(x))

    if is_train:
        start_dt = date(true_pred_year, true_pred_month, 1)
        y = order[pd.date_range(start_dt, periods=1, freq='M')].values.ravel()
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X


# In[39]:


def get_pre_15_days(order, dis, inv, index, year, month):
    X = {}
    start_dt, end_dt = date(year, month, 1), date(year, month, 15)

    # 每个产品M月前10天的提货量
    if order is not None:
        ord_tmp = order.loc[order.order_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
        ord_tmp = ord_tmp.groupby('item_code')[['ord_qty']].sum()
        ord_tmp['ord_qty'] = ord_tmp.ord_qty.apply(lambda x: x if x > 0 else 0)
        ord_tmp = ord_tmp.reindex(index).fillna(0)
        X['ord_pre_15_days'] = ord_tmp.values.ravel()

    # 每个产品M月前10天的分销量
    if dis is not None:
        dis_tmp = dis.loc[dis.dis_date.isin(pd.date_range(start_dt, end_dt, freq='D'))]
        dis_tmp = dis_tmp.groupby('item_code')[['dis_qty']].sum()
        dis_tmp['ord_qty'] = dis_tmp.dis_qty.apply(lambda x: x if x > 0 else 0)
        dis_tmp = dis_tmp.reindex(index).fillna(0)
        X['dis_pre_15_days'] = dis_tmp.values.ravel()

    # 每个产品M月前10天的库存
    if inv is not None:
        inv_tmp = inv.loc[inv.inv_date.isin(pd.date_range(end=end_dt, periods=1, freq='D'))]
        inv_tmp = inv_tmp.groupby('item_code')[['inv_qty']].sum()
        inv_tmp['ord_qty'] = inv_tmp.inv_qty.apply(lambda x: x if x > 0 else 0)
        inv_tmp = inv_tmp.reindex(index).fillna(0)
        X['inv_pre_15_days'] = inv_tmp.values.ravel()

    X = pd.DataFrame(X)

    return X


# In[40]:


def get_pre_15_days_v2(order, dis, inv, year, month, prefix=None):
    X = {}
    start_dt = '%d-%02d-1' % (year, month)
    if order is not None:
        X['ord_pre_15_days'] = order[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()
    if dis is not None:
        X['dis_pre_15_days'] = dis[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()
    if inv is not None:
        X['inv_pre_15_days'] = inv[pd.date_range(start=start_dt, periods=1, freq='M')].values.ravel()
    X = pd.DataFrame(X)
    if prefix:
        X.columns = ['%s_%s' % (prefix, c) for c in X.columns]
    return X


# In[41]:


def prepare_training_set(order, dis, inv,
                         order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                         order_cate1_month, dis_cate1_month, inv_cate1_month,
                         order_cate2_month, dis_cate2_month, inv_cate2_month,
                         sku_info, months, gap):
    X_l, y_l = [], []
    for y_m in months:
        y_str, m_str = y_m.split('-')
        y, m = int(y_str), int(m_str)

        X_tmp, y_tmp = prepare_dataset_v2(order, dis, inv, y, m, gap)
        X_tmp1 = get_pre_15_days_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, y, m)
        #         X_tmp1 = prepare_dataset_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
        #                                     y, m, gap, add_cate_feat=False, is_train=False, name_prefix='sku2')
        X_tmp2 = prepare_dataset_v2(order_cate1_month, dis_cate1_month, inv_cate1_month,
                                    y, m, gap, add_cate_feat=False, is_train=False, name_prefix='cate1')
        X_tmp3 = prepare_dataset_v2(order_cate2_month, dis_cate2_month, inv_cate2_month,
                                    y, m, gap, add_cate_feat=False, is_train=False, name_prefix='cate2')
        X_tmp = pd.concat(
            [X_tmp, X_tmp1, X_tmp2, X_tmp3, sku_info.reset_index(drop=True)], axis=1
        )
        X_l.append(X_tmp)
        y_l.append(y_tmp)
        del X_tmp, y_tmp, X_tmp2, X_tmp3
        gc.collect()

    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)
    return X_train, y_train


# In[42]:


def prepare_val_set(order, dis, inv,
                    order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                    order_cate1_month, dis_cate1_month, inv_cate1_month,
                    order_cate2_month, dis_cate2_month, inv_cate2_month,
                    sku_info, year, month, gap):
    X_val, y_val = prepare_dataset_v2(order, dis, inv, year, month, gap)
    X_val1 = get_pre_15_days_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, year, month)
    #     X_val1 = prepare_dataset_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
    #                                 year, month, gap,
    #                                 add_cate_feat=False, is_train=False, name_prefix='sku2')
    X_val2 = prepare_dataset_v2(order_cate1_month,
                                dis_cate1_month,
                                inv_cate1_month,
                                year, month, gap,
                                add_cate_feat=False, is_train=False, name_prefix='cate1')
    X_val3 = prepare_dataset_v2(order_cate2_month, dis_cate2_month, inv_cate2_month,
                                year, month, gap, add_cate_feat=False, is_train=False, name_prefix='cate2')
    X_val = pd.concat(
        [X_val, X_val1, X_val2, X_val3, sku_info.reset_index(drop=True)],
        axis=1
    )
    del X_val2, X_val3
    gc.collect()
    return X_val, y_val


# In[43]:


def prepare_testing_set(order, dis, inv,
                        order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
                        order_cate1_month, dis_cate1_month, inv_cate1_month,
                        order_cate2_month, dis_cate2_month, inv_cate2_month,
                        sku_info, year, month, gap):
    X_test = prepare_dataset_v2(order, dis, inv, year, month, gap, is_train=False)
    X_test1 = get_pre_15_days_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15, year, month)
    #     X_test1 = prepare_dataset_v2(order_sku_pre_15, dis_sku_pre_15, inv_sku_pre_15,
    #                                  year, month, gap,
    #                                  add_cate_feat=False, is_train=False, name_prefix='sku2')
    X_test2 = prepare_dataset_v2(order_cate1_month, dis_cate1_month, inv_cate1_month,
                                 year, month, gap, add_cate_feat=False, is_train=False, name_prefix='cate1')
    X_test3 = prepare_dataset_v2(order_cate2_month, dis_cate2_month, inv_cate2_month,
                                 year, month, gap, add_cate_feat=False, is_train=False, name_prefix='cate2')
    X_test = pd.concat(
        [X_test, X_test1, X_test2, X_test3, sku_info.reset_index(drop=True)],
        axis=1
    )
    del X_test2, X_test3
    gc.collect()
    return X_test


# # 3 训练和预测

# In[44]:


def get_pre_months(year, month, left_bound='2016-01'):
    start_year, start_month = int(left_bound.split('-')[0]), int(left_bound.split('-')[1])
    all_months = []
    if year == start_year and month >= start_month:
        months = ['%d-%02d' % (start_year, m) for m in range(start_month, month + 1)]
        all_months.extend(months)
    elif year == start_year + 1:
        start_year_months = ['%d-%02d' % (start_year, m) for m in range(start_month, 13)]
        all_months.extend(start_year_months)
        curr_year_months = ['%d-%02d' % (year, m) for m in range(1, month + 1)]
        all_months.extend(curr_year_months)
    elif year > start_year + 1:
        start_year_months = ['%d-%02d' % (start_year, m) for m in range(start_month, 13)]
        all_months.extend(start_year_months)
        for y in range(start_year + 1, year):
            months = ['%d-%02d' % (y, m) for m in range(1, 13)]
            all_months.extend(months)
        curr_year_months = ['%d-%02d' % (year, m) for m in range(1, month + 1)]
        all_months.extend(curr_year_months)
    else:
        raise Exception("[ERROR] The input date is earlier than the start date!!!")
    return all_months


# In[45]:


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


# In[46]:


def modify_training_set(X_train, y_train):
    X_train['y'] = y_train
    X_train = X_train.loc[X_train.y > 0]
    y_train = X_train['y']
    X_train.drop(columns=['y'], inplace=True)
    return X_train, y_train


# In[47]:


year_upper_bound, month_upper_bound = infer_month(start_pred_year,
                                                  start_pred_month,
                                                  offset=-periods)
train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2016-03')
# train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2018-07')

print("[INFO] Start training and predicting...")
t0 = time.time()

MAX_ROUNDS = 500
params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}
cate_vars = []
# cate_vars = ['first_cate_id', 'second_cate_id', 'state']
weight = list(sku_info_filtered['need_calc'] * 2) * len(train_months)

preds_train, preds_test = [], []
mapes_train = []
rmses_train = []
maes_train = []
for i in range(1, periods + 1):
    print()
    print('# ' + '=' * 100 + ' #')
    print('# ' + 'Step %d' % i + ' ' * (100 - len('Step %d' % i)) + ' #')
    print('# ' + '=' * 100 + ' #')

    X_train, y_train = prepare_training_set(order_sku_month, None, None,
                                            order_sku_pre_15, None, None,
                                            order_cate1_month, None, None,
                                            order_cate2_month, None, None,
                                            sku_info_filtered, train_months, gap=i)
    X_train, y_train = modify_training_set(X_train, y_train)
    X_test = prepare_testing_set(order_sku_month, None, None,
                                 order_sku_pre_15, None, None,
                                 order_cate1_month, None, None,
                                 order_cate2_month, None, None,
                                 sku_info_filtered, start_pred_year, start_pred_month, gap=i)

    # Add previous predictions as a new feature
    #     if preds_train:
    #         X_train['m%s' % (i - 1)] = pd.Series(preds_train[i - 1])
    #         X_val['m%s' % (i - 1)] = pd.Series(preds_train[i - 1])
    #         X_test['m%s' % (i - 1)] = pd.Series(preds_test[i - 1])

    # Adjust the month predicted
    #     if i != 0:
    #         X_train['pred_month'] = X_train.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)
    #         X_test['pred_month'] = X_test.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cate_vars)

    print("[INFO] Fit the model...")
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain], early_stopping_rounds=125, verbose_eval=50
    )

    # Predict
    pred_train = bst.predict(X_train, num_iteration=bst.best_iteration or MAX_ROUNDS)
    pred_test = bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS)

    # Calculate accuracy
    #     acc_train = acc(y_train, pred_train)
    #     print("[INFO] acc_train: %.4f" % (acc_train))

    #     acc_v2_train = acc_v2(y_train, pred_train)
    #     print("[INFO] acc_v2_train: %.4f" % (acc_v2_train))

    # Calculate MAPE
    mape_train = mean_absolute_percent_error(y_train, pred_train)
    print("[INFO] mape_train: %.4f" % (mape_train))

    # Calculate RMSE
    rmse_train = sqrt(mean_squared_error(y_train, pred_train))
    print("[INFO] rmse_train: %.4f" % (rmse_train))

    # Calculate MAE
    #     mae_train = mean_absolute_error(y_train, pred_train)
    #     print("[INFO] mae_train: %.4f" % (mae_train))

    # Store the intermediate results
    preds_train.append(pred_train)
    preds_test.append(pred_test)
    #     mapes_train.append(mape_train)
    #     rmses_train.append(rmse_train)
    #     maes_train.append(mae_train)

    # Output feature importances
    feat_imps = sorted(zip(X_train.columns, bst.feature_importance('gain')),
                       key=lambda x: x[1], reverse=True)
    print("The feature importances are as follow: ")
    print('\n'.join('%s: %s' % (feat_name, feat_imp) for feat_name, feat_imp in feat_imps))

print()
print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time.time() - t0))

# In[48]:


X_train.shape


# # 4 结果评估

# In[49]:


def timestamp_to_time(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


# In[50]:


def add_accuracy(df, acc_col_name, act_val_col, pred_val_col):
    df[acc_col_name] = df[act_val_col] / df[pred_val_col]
    df[acc_col_name] = df[acc_col_name].apply(lambda x: 1 / x if x > 1 else x)


# In[51]:


column_names = ['pred_ord_qty_m%d' % i for i in range(1, periods + 1)]
result = pd.DataFrame(
    np.array(preds_test).transpose(), index=order_sku_month.index,
    columns=column_names
)
# result = np.round(result * 10000)
for col in result.columns:
    result[col] = result[col].apply(lambda x: 0.0025 if x < 0 else x)
result = result.reset_index()

# In[52]:


result['comb_name'] = 'default'
result['bu_code'] = '30015305'
result['bu_name'] = '厨房热水器事业部'
result['order_date'] = '%d-%02d-%d' % (start_pred_year,
                                       start_pred_month,
                                       get_days_of_month(start_pred_year, start_pred_month))
result['item_name'] = result.item_code.map(sku_info.to_dict()['item_name'])
result['first_cate_code'] = result.item_code.map(sku_info.to_dict()['first_cate_code'])
result['second_cate_code'] = result.item_code.map(sku_info.to_dict()['second_cate_code'])
result['first_cate_name'] = result.item_code.map(sku_info.to_dict()['first_cate_name'])
result['second_cate_name'] = result.item_code.map(sku_info.to_dict()['second_cate_name'])
result['item_price'] = result.item_code.map(sku_info.to_dict()['item_price'])
result['sales_type'] = '内销'
result['forecast_type'] = '内销整机预测'
result['manu_code'] = result.item_code.map(items.set_index('item_code').to_dict()['manu_code'])
result.manu_code.fillna('', inplace=True)
result['area_name'] = ''
result['ord_pred_time'] = timestamp_to_time(time.time())

# ## 4.1 加入规则

# In[53]:


plan_path = "/data/aidev/order-sales-forecast_old/data/plan/%s/m111-plan_%s.txt" % tuple([data_file_flag] * 2)
column_names = [
    'bd_code',
    'bd_name',
    'order_type_name',
    'period_code',
    'begin_date',
    'end_date',
    'customer_code',
    'customer_name',
    'sales_center_code',
    'sales_center_name',
    'item_code',
    'item_name',
    'plan_mode',
    'month_plan_qty',
    'm2_month_plan_qty',
    'm3_month_plan_qty',
    'w_insert_dt'
]
plan = pd.read_csv(plan_path, sep='\u001e', header=None, names=column_names)
plan = plan.loc[plan.bd_code == 'M111']
plan = plan.loc[plan.plan_mode.isin(['NORMAL', 'CUSTOMER'])]
plan['item_code'] = plan.item_code.astype(str).apply(lambda x: x.strip())
plan['period_code'] = pd.to_datetime(plan.period_code, format='%Y%m')
plan = plan.sort_values(by=['period_code'], ascending=True)
m1_year, m1_month = infer_month(curr_year, curr_month, 1)
plan = plan.loc[plan.period_code <= '%d-%02d-01' % (m1_year, m1_month)]
plan['month_plan_qty'] = plan.month_plan_qty / 10000

# In[54]:


# 每个SKU每个月的需求数
temp = plan.copy()
temp['order_date'] = temp.period_code.astype(str).apply(lambda x: x[:7])
temp = temp.groupby(['item_code', 'order_date'])[['month_plan_qty']].sum()
plan_sku_month = temp.unstack(level=-1).fillna(0)
end_dt_str = '%d-%02d-%d' % (m1_year,
                             m1_month,
                             get_days_of_month(m1_year, m1_month))
plan_sku_month.columns = pd.date_range(start='2018-12-31', end=end_dt_str, freq='M')
plan_sku_month = plan_sku_month.reindex(order_sku_month.index).fillna(0)

del temp
gc.collect()


# In[55]:


# 从文件读取，弃用

# 需求数据
# plan_filepath = "../../data/opt/m111-plan-%d%02d.txt" % (curr_year, curr_month)
# plan_filepath = "../../data/plan/m111-plan-%d%02d.txt" % (2019, 11)
# plan = pd.read_csv(plan_filepath, sep='\t')
# plan['item_code'] = plan.item_code.astype(str).apply(lambda x: x.strip())
# plan['period_code'] = pd.to_datetime(plan.period_code, format='%Y%m')
# plan = plan.sort_values(by=['period_code'], ascending=True)
# plan = plan.loc[plan.plan_mode.isin(['NORMAL', 'CUSTOMER'])]
# plan = plan.loc[plan.period_code <= '%d-%02d-01' % get_month(curr_year, curr_month, 1)]
# plan['month_plan_qty'] = plan.month_plan_qty / 10000

# 每个SKU每个月的需求数
# temp = plan.copy()
# temp['order_date'] = temp.period_code.astype(str).apply(lambda x: x[:7])
# temp = temp.groupby(['item_code', 'order_date'])[['month_plan_qty']].sum()
# plan_sku_month = temp.unstack(level=-1).fillna(0)
# end_dt_str = '%d-%02d-%d' % (curr_year, 
#                              curr_month, 
#                              get_days_of_month(curr_year, curr_month))
# plan_sku_month.columns = pd.date_range(start='2019-01-31', end=end_dt_str, freq='M')
# plan_sku_month = plan_sku_month.reindex(order_sku_month.index).fillna(0)

# del temp
# gc.collect()

# 需求数据2月异常，弃用
# plan_sku_month.iloc[:, 2] = 0


# In[56]:


def _get_pre_vals(df: pd.DataFrame, year: int, month: int, periods: int = 3) -> pd.DataFrame:
    end_dt = date(year, month, 1)
    return df[pd.date_range(end=end_dt, periods=periods, freq='M')]


# In[57]:


def _get_val(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    start_dt = date(year, month, 1)
    return df[pd.date_range(start=start_dt, periods=1, freq='M')].iloc[:, 0]


# In[58]:


order_sku_month_pre6_mean = _get_pre_vals(
    order_sku_month, start_pred_year, start_pred_month, 6
).replace(0, np.nan).mean(axis=1)

# In[59]:


dis_sku_month_pre3_mean = _get_pre_vals(
    dis_sku_month, start_pred_year, start_pred_month, 3
).replace(0, np.nan).mean(axis=1)

# In[60]:


plan_sku_month_mean = plan_sku_month.replace(0, np.nan).mean(axis=1)

# In[61]:


pre1_year, pre1_month = infer_month(start_pred_year, start_pred_month, -1)
order_sku_month_pre1 = _get_val(order_sku_month, pre1_year, pre1_month).replace(0, np.nan)
dis_sku_month_pre1 = _get_val(dis_sku_month, pre1_year, pre1_month).replace(0, np.nan)

# In[62]:


tmp = result[['item_code', 'first_cate_name', 'pred_ord_qty_m1']].rename(
    columns={'pred_ord_qty_m1': 'pred_ord_qty'}
)

# In[63]:


tmp['ord_sku_month_pre6_mean'] = tmp.item_code.map(order_sku_month_pre6_mean)
tmp['dis_sku_month_pre3_mean'] = tmp.item_code.map(dis_sku_month_pre3_mean)
tmp['plan_sku_month_mean'] = tmp.item_code.map(plan_sku_month_mean)
tmp['ord_sku_month_pre1'] = tmp.item_code.map(order_sku_month_pre1)
tmp['dis_sku_month_pre1'] = tmp.item_code.map(dis_sku_month_pre1)

# In[64]:


tmp['is_aver_ord_na'] = (tmp.ord_sku_month_pre6_mean.isna()) * 1
tmp['is_aver_dis_na'] = (tmp.dis_sku_month_pre3_mean.isna()) * 1
tmp['is_aver_plan_na'] = (tmp.plan_sku_month_mean.isna()) * 1
tmp['is_ord_pre1_na'] = (tmp.ord_sku_month_pre1.isna()) * 1
tmp['is_dis_pre1_na'] = (tmp.dis_sku_month_pre1.isna()) * 1

# In[65]:


# temp1 = get_val(order_sku_month, 2018, 10).to_dict()
# temp2 = get_val(order_sku_month, 2018, 9).to_dict()
# tmp['temp1'] = tmp.item_code.map(temp1)
# tmp['temp2'] = tmp.item_code.map(temp2)
tmp['online_offline_flag'] = tmp.item_code.map(sku_info.to_dict()['sales_chan_name']).fillna('未知')
tmp['project_flag'] = tmp.item_code.map(sku_info.to_dict()['project_flag']).fillna('未知')

# In[66]:


# 单月新品
temp = _get_pre_vals(
    order_sku_month, start_pred_year, start_pred_month, 24
).replace(0, np.nan).mean(axis=1)
curr_new_items = set(temp.loc[temp.isna()].index)

del temp
gc.collect()

# In[67]:


# 近三个月中只有最近一个有分销的SKU
temp = _get_pre_vals(
    dis_sku_month, start_pred_year, start_pred_month, 3
)
temp['num_not_null'] = ((temp > 0) * 1).sum(axis=1)
new_items_by_dis = set(temp.loc[(temp.num_not_null == 1) & (temp.iloc[:, 2] > 0)].index)

del temp
gc.collect()

# In[68]:


temp_year, temp_month = infer_month(start_pred_year, start_pred_month, 1)
temp = _get_val(plan_sku_month, temp_year, temp_month)
tmp['demand'] = tmp.item_code.map(temp)
tmp['is_curr_new'] = tmp.item_code.apply(lambda x: 1 if x in curr_new_items else 0)
tmp['is_new_by_dis'] = tmp.item_code.apply(lambda x: 1 if x in new_items_by_dis else 0)
tmp['demand_dis_ratio'] = tmp.demand / tmp.dis_sku_month_pre3_mean


# In[69]:


def rule_func(df):
    if df.first_cate_name in {'净水机', '饮水机', '洗碗机', '消毒柜'}:
        if df.is_curr_new:  # 如果是当月新品
            return df.demand
        else:  # 如果不是当月新品
            if df.is_dis_pre1_na or df.is_aver_dis_na:  # 如果上月分销或者月均分销为空
                if df.demand == 0:
                    return df.pred_ord_qty
                else:
                    return df.demand
            else:  # 如果上月分销或者月均分销不为空
                if df.demand == 0:
                    return df.pred_ord_qty
                else:
                    if df.demand_dis_ratio > 1.1:
                        return (df.demand + df.dis_sku_month_pre3_mean) / 2
                    else:
                        if df.demand > df.dis_sku_month_pre1:
                            return (df.demand + df.dis_sku_month_pre3_mean) / 2
                        else:
                            return df.demand
    elif df.first_cate_name in {'烟机', '灶具'}:
        if df.project_flag == '是' or df.is_new_by_dis:
            return (df.dis_sku_month_pre3_mean + df.demand) / 2
        else:
            if df.is_aver_dis_na:
                if df.is_aver_plan_na:
                    if df.demand == 0:
                        return df.pred_ord_qty
                    else:
                        return df.demand
                else:
                    return (df.demand + df.plan_sku_month_mean) / 2
            else:
                if df.demand == 0:
                    if df.is_aver_ord_na:
                        if df.is_aver_plan_na:
                            return df.demand
                        else:
                            return df.plan_sku_month_mean
                    else:
                        return df.ord_sku_month_pre6_mean
                else:
                    if df.demand >= 10000:
                        return (df.ord_sku_month_pre1 + df.dis_sku_month_pre1) / 2
                    else:
                        return (df.demand + df.ord_sku_month_pre6_mean) / 2
    elif df.first_cate_name in {'电热水器', '燃气热水器'}:
        if df.demand == 0:
            if df.is_aver_plan_na:
                return df.pred_ord_qty
            else:
                return df.plan_sku_month_mean
        else:
            if df.is_aver_ord_na:
                if df.is_aver_plan_na:
                    return df.demand
                else:
                    return df.plan_sku_month_mean
            else:
                if df.first_cate_name == '电热水器':
                    if df.demand >= 10000:
                        return (df.dis_sku_month_pre1 + df.demand) / 2
                    else:
                        return (df.demand + df.ord_sku_month_pre6_mean) / 2
                elif df.first_cate_name == '燃气热水器':
                    return (df.demand + df.ord_sku_month_pre6_mean) / 2
                else:
                    return df.pred_ord_qty
    else:
        return df.pred_ord_qty


# In[70]:


def rule_func_v1(df):
    if df.first_cate_name in {'净水机', '饮水机', '洗碗机', '消毒柜'}:
        if df.is_curr_new:  # 如果是当月新品
            return df.demand
        else:  # 如果不是当月新品
            if df.is_dis_pre1_na or df.is_aver_dis_na:  # 如果上月分销或者月均分销为空
                if df.demand == 0:
                    return df.pred_ord_qty
                else:
                    return df.demand
            else:  # 如果上月分销或者月均分销不为空
                if df.demand == 0:
                    return df.pred_ord_qty
                else:
                    if df.demand_dis_ratio > 1.1:
                        return (df.demand + df.dis_sku_month_pre3_mean) / 2
                    else:
                        if df.demand > df.dis_sku_month_pre1:
                            return (df.demand + df.dis_sku_month_pre3_mean) / 2
                        else:
                            return df.demand
    elif df.first_cate_name in {'烟机', '灶具'}:
        if df.project_flag == '是':
            return (df.dis_sku_month_pre3_mean + df.demand) / 2
        elif df.is_new_by_dis:
            return df.ord_sku_month_pre1
        else:
            if df.is_aver_dis_na:
                if df.is_aver_plan_na:
                    if df.demand == 0:
                        return df.pred_ord_qty
                    else:
                        return df.demand
                else:
                    return (df.demand + df.plan_sku_month_mean) / 2
            else:
                if df.demand == 0:
                    if df.is_aver_ord_na:
                        if df.is_aver_plan_na:
                            return df.demand
                        else:
                            return df.plan_sku_month_mean
                    else:
                        return df.ord_sku_month_pre6_mean
                else:
                    return (df.demand + df.plan_sku_month_mean) / 2
    elif df.first_cate_name in {'电热水器', '燃气热水器'}:
        if df.demand == 0:
            if df.is_aver_plan_na:
                return df.pred_ord_qty
            else:
                return df.plan_sku_month_mean
        else:
            if df.is_aver_ord_na:
                if df.is_aver_plan_na:
                    return df.demand
                else:
                    return df.plan_sku_month_mean
            else:
                if df.first_cate_name == '电热水器':
                    if df.demand >= 10000 / 10000:
                        return (df.dis_sku_month_pre1 + df.demand) / 2
                    else:
                        return (df.demand + df.ord_sku_month_pre6_mean) / 2
                elif df.first_cate_name == '燃气热水器':
                    return (df.demand + df.ord_sku_month_pre6_mean) / 2
                else:
                    return df.pred_ord_qty
    else:
        return df.pred_ord_qty


# In[71]:


tmp['pred_ord_qty_rule'] = tmp.apply(rule_func_v1, axis=1)
tmp['pred_ord_qty_rule'] = tmp.pred_ord_qty_rule.replace(np.nan, 0)
tmp['pred_ord_qty_rule'] = tmp.apply(
    lambda x: x.pred_ord_qty if x.pred_ord_qty_rule == 0 else x.pred_ord_qty_rule,
    axis=1
)


# In[72]:


# # M0月前15天的提货量
# temp = get_val(order_sku_pre_15, start_pred_year, start_pred_month).to_dict()
# m1_res['ord_sku_pre_15'] = m1_res.item_code.map(temp)

# # M0月实际值
# temp = get_val(order_sku_month, start_pred_year, start_pred_month).to_dict()
# m1_res['m0_ord'] = m1_res.item_code.map(temp)

# # 前1个月的提货
# pre1_year, pre1_month = get_month(start_pred_year, start_pred_month, -1)
# temp = get_val(order_sku_month, pre1_year, pre1_month)
# m1_res['ord_pre1'] = m1_res.item_code.map(temp)

# # 前2个月的提货
# pre2_year, pre2_month = get_month(start_pred_year, start_pred_month, -2)
# temp = get_val(order_sku_month, pre2_year, pre2_month)
# m1_res['ord_pre2'] = m1_res.item_code.map(temp)

# # 前3个月的提货
# pre3_year, pre3_month = get_month(start_pred_year, start_pred_month, -3)
# temp = get_val(order_sku_month, pre3_year, pre3_month)
# m1_res['ord_pre3'] = m1_res.item_code.map(temp)


# In[73]:


# m1_res = m1_res[['order_date', 
#                  'item_code', 
#                  'item_name', 
#                  'first_cate_name', 
#                  'second_cate_name', 
#                  'item_price', 
#                  'act_ord_qty', 
#                  'pred_ord_qty', 
#                  'dis_sku_month_pre3_mean', 
#                  'dis_sku_month_pre1',
# #                  'temp1', 
# #                  'temp2', 
#                  'model_ord_acc', 
#                  'model_ord_weighted_acc', 
#                  'online_offline_flag',
#                  'project_flag']]


# In[74]:


# m1_res = m1_res[['order_date', 
#                  'item_code', 
#                  'item_name', 
#                  'first_cate_name', 
#                  'second_cate_name', 
#                  'project_flag', 
#                  'online_offline_flag', 
#                  'item_price', 
#                  'ord_sku_month_pre1', 
#                  'dis_sku_month_pre1', 
#                  'ord_sku_month_pre6_mean', 
#                  'dis_sku_month_pre3_mean', 
#                  'plan_sku_month_mean', 
#                  'demand', 
#                  'pred_ord_qty', 
#                  'pred_ord_qty_rule', 
#                  'act_ord_qty', 
#                  'model_ord_acc', 
#                  'model_ord_weighted_acc', 
#                  'rule_ord_acc', 
#                  'rule_ord_weighted_acc']]


# In[75]:


# result_root_dir = "../../mid_results/level2/test/"
# if not os.path.exists(result_root_dir):
#     os.makedirs(result_root_dir)
# filename = "level2_res_%d%02d.txt" % (true_pred_year, 
#                                       true_pred_month)
# result_path = os.path.join(result_root_dir, filename)
# if os.path.exists(result_path):
#     os.remove(result_path)
# m1_res.to_csv(result_path, sep=',', index=None)


# ## 4.2 模型与规则融合

# In[76]:


def ensemble_rule(df):
    if df.first_cate_name in {'电热水器', '燃气热水器'}:
        return df.pred_ord_qty_rule
    else:
        return df.pred_ord_qty * 0.5 + df.pred_ord_qty_rule * 0.5


tmp['ensemble'] = tmp.apply(ensemble_rule, axis=1)
result['pred_ord_qty_m1'] = tmp.ensemble
result['avg_dis'] = np.round(tmp['dis_sku_month_pre3_mean'] * 10000)

# In[77]:


result['pred_ord_qty_m1'] = np.round(result.pred_ord_qty_m1 * 10000)
result['pred_ord_qty_m2'] = np.round(result.pred_ord_qty_m2 * 10000)
result['pred_ord_qty_m3'] = np.round(result.pred_ord_qty_m3 * 10000)

for i in range(1, periods + 1):
    result['pred_ord_amount_m%d' % i] = result['pred_ord_qty_m%d' % i] * result.item_price

# In[78]:


result = result.loc[~result.item_code.isin(delisting_items)]
result = result.loc[~(result.manu_code == '')]
result['avg_dis'] = result.avg_dis.fillna(0.0)

# In[79]:


result = result[['bu_code',
                 'order_date',
                 'item_code',
                 'first_cate_code',
                 'second_cate_code',
                 'comb_name',
                 'bu_name',
                 'item_name',
                 'first_cate_name',
                 'second_cate_name',
                 'item_price',
                 'sales_type',
                 'forecast_type',
                 'manu_code',
                 'area_name',
                 'avg_dis',
                 'ord_pred_time',
                 'pred_ord_qty_m1',
                 'pred_ord_amount_m1',
                 'pred_ord_qty_m2',
                 'pred_ord_amount_m2',
                 'pred_ord_qty_m3',
                 'pred_ord_amount_m3']]

# In[80]:


# m1_year, m1_month = get_month(start_pred_year, start_pred_month, 1)
# m2_year, m2_month = get_month(start_pred_year, start_pred_month, 2)
# m3_year, m3_month = get_month(start_pred_year, start_pred_month, 3)


# In[81]:


# m1_act = np.round(_get_val(order_sku_month, m1_year, m1_month) * 10000)
# m2_act = np.round(_get_val(order_sku_month, m2_year, m2_month) * 10000)
# m3_act = np.round(_get_val(order_sku_month, m3_year, m3_month) * 10000)


# In[82]:


# result['act_ord_qty_m1'] = result.item_code.map(m1_act)
# result['act_ord_qty_m2'] = result.item_code.map(m2_act)
# result['act_ord_qty_m3'] = result.item_code.map(m3_act)


# In[99]:


result_root_dir = "/data/aidev/order-sales-forecast_old/mid_results/level2/order/%d-%02d" % (curr_year, curr_month)
if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)
filename = "level2_order_result_%d%02d.txt" % (start_pred_year,
                                               start_pred_month)
result_path = os.path.join(result_root_dir, filename)
if os.path.exists(result_path):
    os.remove(result_path)
result.to_csv(result_path, sep=',', index=None)

# In[84]:


# 删除不需要计算的SKU和当月新品
# result = result.loc[result.item_code.isin(white_items)]
# result = result.loc[~result.item_code.isin(curr_new_items)]
# print("The average accuracy of all categories is: %.4f" % (result.ord_acc.mean()))
# print("The weighted accuracy of all categories is: %.4f" % (result.ord_weighted_acc.sum() / result.act_ord_qty.sum()))


# In[85]:


# 处理预测低于30%的情况（用当月前15天的提货量的两倍作为实际值）
# start_dt = '%d-%02d-1' % (start_pred_year, start_pred_month)
# dummy_values = order_sku_pre_15[pd.date_range(start=start_dt, periods=1, freq='M')].iloc[:, 0].to_dict()
# result['dummy_ord_qty'] = result.item_code.map(dummy_values).fillna(0)
# add_accuracy(result, 'dummy_ord_acc', 'dummy_ord_qty', 'pred_ord_qty')

# result['flag'] = result.apply(lambda x: 1 if x.ord_acc <= 0.3 and x.dummy_ord_acc <= 0.3 else 0, axis=1)
# print(len(result.loc[result.flag == 1]) / len(result.loc[result.ord_acc <= 0.3]))
# print(len(result.loc[result.flag == 1]) / len(result.loc[result.dummy_ord_acc <= 0.3]))

# result['pred_ord_qty'] = result.apply(
#     lambda x: x.dummy_ord_qty if x.dummy_ord_acc <= 0.3 else x.pred_ord_qty, axis=1
# )
# add_accuracy(result, 'ord_acc', 'act_ord_qty', 'pred_ord_qty')
# result['ord_weighted_acc'] = (result.act_ord_qty * result.ord_acc).astype(np.float32)
# print("The average accuracy of all categories is: %.4f" % (result.ord_acc.mean()))
# print("The weighted accuracy of all categories is: %.4f" % (result.ord_weighted_acc.sum() / result.act_ord_qty.sum()))


# In[86]:


# 处理预测值为0的情况
# pred_zero_items = set(result.loc[result.pred_ord_qty == 0].item_code.unique())
# start_dt = '%d-%02d-01' % (start_pred_year, start_pred_month)
# replace_values = order_sku_pre_15.loc[pred_zero_items][pd.date_range(start=start_dt, periods=1, freq='M')]
# replace_values = replace_values.iloc[:, 0].to_dict()
# result['new_pred_ord_qty'] = result.item_code.map(replace_values).fillna(0)
# result['pred_ord_qty'] = result.pred_ord_qty + result.new_pred_ord_qty
# result.drop(columns=['new_pred_ord_qty'], inplace=True)
# result['pred_ord_qty'] = result.pred_ord_qty.apply(lambda x: 50 if x == 0 else x)


# In[87]:


# add_accuracy(result, 'ord_acc', 'act_ord_qty', 'pred_ord_qty')
# result['ord_weighted_acc'] = (result.act_ord_qty * result.ord_acc).astype(np.float32)
# print("The average accuracy of all categories is: %.4f" % (result.ord_acc.mean()))
# print("The weighted accuracy of all categories is: %.4f" % (result.ord_weighted_acc.sum() / result.act_ord_qty.sum()))


# In[88]:


# yj_res = result.loc[result.first_cate_code == 'CRYJ']
# zj_res = result.loc[result.first_cate_code == 'CRZJ']
# xdg_res = result.loc[result.first_cate_code == 'CRXDG']
# xwj_res = result.loc[result.first_cate_code == 'CRXWJ']

# dr_res = result.loc[result.first_cate_code == 'DR']
# rr_res = result.loc[result.first_cate_code == 'RR']
# ysj_res = result.loc[result.first_cate_code == 'YSJ']
# jsj_res = result.loc[result.first_cate_code == 'JSJ']


# In[89]:


# yj_weighted_acc = yj_res.ord_weighted_acc.sum() / yj_res.act_ord_qty.sum()
# zj_weighted_acc = zj_res.ord_weighted_acc.sum() / zj_res.act_ord_qty.sum()
# xdg_weighted_acc = xdg_res.ord_weighted_acc.sum() / xdg_res.act_ord_qty.sum()
# xwj_weighted_acc = xwj_res.ord_weighted_acc.sum() / xwj_res.act_ord_qty.sum()

# dr_weighted_acc = dr_res.ord_weighted_acc.sum() / dr_res.act_ord_qty.sum()
# rr_weighted_acc = rr_res.ord_weighted_acc.sum() / rr_res.act_ord_qty.sum()
# ysj_weighted_acc = ysj_res.ord_weighted_acc.sum() / ysj_res.act_ord_qty.sum()
# jsj_weighted_acc = jsj_res.ord_weighted_acc.sum() / jsj_res.act_ord_qty.sum()

# print("The weighted accuracy of 'yanji' is: %.4f" % (yj_weighted_acc))
# print("The weighted accuracy of 'zaoju' is: %.4f" % (zj_weighted_acc))
# print("The weighted accuracy of 'xiaodugui' is: %.4f" % (xdg_weighted_acc))
# print("The weighted accuracy of 'xiwanji' is: %.4f" % (xwj_weighted_acc))

# print()

# print("The weighted accuracy of 'dianre' is: %.4f" % (dr_weighted_acc))
# print("The weighted accuracy of 'ranre' is: %.4f" % (rr_weighted_acc))
# print("The weighted accuracy of 'yinshuiji' is: %.4f" % (ysj_weighted_acc))
# print("The weighted accuracy of 'jingshuiji' is: %.4f" % (jsj_weighted_acc))


# ## 5 结果写入

# In[90]:


# 得到缩放比例
# level1_result = pd.read_csv(
#     "level1-result_2019-05-14.txt", dtype={'order_date': str}
# ).rename(columns={'pred_ord_qty': 'level1_pred_ord_qty'})
# level1_result = level1_result[['first_cate_code', 'order_date', 'level1_pred_ord_qty']]

# tmp = df_pred_test.copy()
# tmp['first_cate_code'] = tmp.item_code.map(sku_info.to_dict()['first_cate_code'])
# tmp = tmp.groupby(['first_cate_code', 'order_date'])[['pred_ord_qty']].sum()
# tmp = tmp.join(level1_result.set_index(['first_cate_code', 'order_date']), how='left')
# tmp['scale_factor'] = tmp.level1_pred_ord_qty / tmp.pred_ord_qty
# scale_factor = tmp[['scale_factor']]

# del tmp
# gc.collect()


# In[91]:


# 进行缩放
# tmp = df_pred_test.copy().reset_index()
# tmp['first_cate_code'] = tmp.item_code.map(sku_info.to_dict()['first_cate_code'])
# tmp = tmp.join(scale_factor, on=['first_cate_code', 'order_date'], how='left')
# tmp['pred_ord_qty'] = tmp.pred_ord_qty * tmp.scale_factor
# df_pred_test = tmp.drop(columns=['first_cate_code', 'scale_factor'])
# df_pred_test['pred_ord_qty'] = np.round(df_pred_test.pred_ord_qty, decimals=4)

# del tmp
# gc.collect()


# In[92]:


# feat_desc_path = "../../feat_desc.txt"
# fc_to_fn, fc_to_ft = {}, {}
# with open(feat_desc_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         fc, fn, ft = line.rstrip('\r\n').split('\t')
#         fc_to_fn[fc] = fn
#         fc_to_ft[fc] = ft


# In[93]:


# imp = pd.DataFrame(feat_imps, columns=['feat_code', 'feat_imp'])
# imp['feat_name'] = imp.feat_code.map(fc_to_fn)
# imp['feat_type'] = imp.feat_code.map(fc_to_ft)
# imp['feat_imp'] = imp.feat_imp / imp.feat_imp.sum()
# imp['pred_time'] = "%d%02d" % (true_pred_year, true_pred_month)
# imp['comb_name'] = 'Default'


# In[94]:


# Save result to TXT file
# result_root_dir = "/data/aidev/order-sales-forecast_old/mid_results/level2/order/%d-%02d" % (curr_year, curr_month)
# if not os.path.exists(result_root_dir):
#     os.makedirs(result_root_dir)
# filename = "level2_order_result_%s_%d%02d_%s.txt" % ('lgb', 
#                                                      true_pred_year, 
#                                                      true_pred_month, 
#                                                      int(time.time()))
# result_path = os.path.join(result_root_dir, filename)
# if os.path.exists(result_path):
#     os.remove(result_path)
# result.to_csv(result_path, sep=',', index=None)


# In[95]:


# Save imp to TXT file
# imp_root_dir = "/data/aidev/order-sales-forecast/mid_results/level2/imp/%d-%02d" % (curr_year, curr_month)
# if not os.path.exists(imp_root_dir):
#     os.makedirs(imp_root_dir)
# filename = "level2_order_imp_%d%02d.txt" % (true_pred_year, true_pred_month)
# imp_path = os.path.join(imp_root_dir, filename)
# if os.path.exists(imp_path):
#     os.remove(imp_path)
# imp.to_csv(imp_path, sep=',', index=None)


# In[96]:


# result['sales_cen_code'] = result.sales_cen_code.astype(str)
# result['customer_code'] = result.customer_code.astype(str)
# result['item_code'] = result.item_code.astype(str)


# In[97]:


# Upsert Kudu database
from impala.dbapi import connect


class SitDbConfig:
    def __init__(self):
        self.host = '10.18.25.92'
        self.port = 31051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


class UatDbConfig:
    def __init__(self):
        self.host = '10.18.25.92'
        self.port = 21051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


class ProdDbConfig:
    def __init__(self):
        self.host = '10.18.25.72'
        self.port = 21051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


def create_cursor(config):
    conn = connect(host=config.host,
                   port=config.port,
                   database=config.database,
                   auth_mechanism=config.auth_mechanism,
                   user=config.user,
                   password=config.password)
    return conn.cursor()


def batch_upsert_data(cursor, df, table_name, batch_size=5000):
    print("[INFO] Start upsert data...")
    t0 = time.time()

    l = len(df)  # the size of data
    n = (l - 1) // batch_size + 1  # number of times to write
    header_str = str(tuple(df.columns)).replace("\'", '')
    for i in range(n):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, l)
        values = [str(tuple(row)) for row in df[start_index:end_index].values]
        values_str = ', '.join(values)
        sql = "UPSERT INTO %s %s VALUES %s;" % (table_name, header_str, values_str)
        try:
            cursor.execute(sql)
        except Exception:
            print("[ERROR] Upsert failed!!!")
            return

    print("[INFO] Upsert finished! ( ^ _ ^ )V")
    print("[INFO] Done in %s seconds." % (time.time() - t0))


# In[1157]:


# db_config = UatDbConfig()
# cursor = create_cursor(db_config)
# cursor.execute("DELETE FROM level2_sku_pred_result WHERE order_date = '%d%02d'" % (true_pred_year, true_pred_month))


# In[1158]:


# db_config = UatDbConfig()
# cursor = create_cursor(db_config)
# result_table_name = 'level2_sku_pred_result'
# batch_upsert_data(cursor, result, result_table_name)


# In[1159]:


# db_config = ProdDbConfig()
# cursor = create_cursor(db_config)
# cursor.execute("DELETE FROM level2_sku_pred_result WHERE order_date = '%d%02d'" % (true_pred_year, true_pred_month))


# In[1160]:


# db_config = ProdDbConfig()
# cursor = create_cursor(db_config)
# result_table_name = 'level2_sku_pred_result'
# batch_upsert_data(cursor, result, result_table_name)


# In[1161]:


# imp_table_name = 'level2_sku_pred_feat_importances'
# batch_upsert_data(cursor, imp, imp_table_name)


# In[1162]:


# db_config = UatDbConfig()
# cursor = create_cursor(db_config)
# cursor.execute('DELETE FROM level2_sku_pred_result')
# cursor.execute('DELETE FROM level2_sku_pred_feat_importances')


# In[100]:


db_config = UatDbConfig()
cursor = create_cursor(db_config)
result_table_name = 'm111_order_predict_res'
batch_upsert_data(cursor, result, result_table_name)

# In[101]:


db_config = ProdDbConfig()
cursor = create_cursor(db_config)
result_table_name = 'm111_order_predict_res'
batch_upsert_data(cursor, result, result_table_name)


# In[102]:


# db_config = ProdDbConfig()
# cursor = create_cursor(db_config)
# result_table_name = 'level2_sku_pred_result'
# batch_upsert_data(cursor, result, result_table_name)


# ## 6 推送给ESB

# In[103]:


def _generate_serial_number():
    dt_str = "%d%02d%02d" % get_curr_date()
    return "m111-order-%s%s" % (dt_str, str(time.time())[-4:])


# In[104]:


def push_data(df, serial_no, url):
    df_json_str = df.to_json(orient='records', force_ascii=False)
    df_json_str = "{\"hData\":%s}" % df_json_str
    json_obj = {
        "Envelope": {
            "Header": {
                "requestHeader": {
                    "version": "1.0",
                    "serialNo": serial_no,
                    "requestId": "BDOF",
                    "namespace": "http://www.midea.com/afp/AfpMassDataImportService/v1"
                }
            },
            "Body": {
                "DataImport": {
                    "serialNo": serial_no,
                    "iFaceCode": "GAPS-BDOF-001",
                    "source_code": "BDOF",
                    "data": df_json_str
                }
            }
        }
    }
    json_str = json.dumps(json_obj)
    response = requests.post(url, data=json_str)
    response_obj = json.loads(response.text)
    if response_obj["Envelope"]["Body"]["DataImportResponse"]["isSuccess"]:
        return response_obj, True
    else:
        return response_obj, False


# In[105]:


result['customer_code'] = ''
result['attribute1'] = ''
result['attribute2'] = ''
result['attribute3'] = ''
result['attribute4'] = ''
result['attribute5'] = ''
result.rename(columns={'manu_code': 'manu_name'}, inplace=True)
result = result[['bu_code',
                 'sales_type',
                 'manu_name',
                 'area_name',
                 'customer_code',
                 'order_date',
                 'first_cate_name',
                 'second_cate_name',
                 'item_code',
                 'forecast_type',
                 'avg_dis',
                 'item_price',
                 'pred_ord_qty_m1',
                 'pred_ord_qty_m2',
                 'pred_ord_qty_m3',
                 'attribute1',
                 'attribute2',
                 'attribute3',
                 'attribute4',
                 'attribute5']]

# In[106]:


result['order_date'] = '%d-%02d-%d' % (true_pred_year,
                                       true_pred_month,
                                       get_days_of_month(true_pred_year, true_pred_month))

# In[107]:


serial_no = _generate_serial_number()
uat_url = "http://10.16.41.76:7801/transactREST"
prod_url = "http://10.18.1.116:7801/transactREST"
try_time, success_mark = 0, False
while not success_mark:
    time.sleep(5)
    _, success_mark = push_data(result, serial_no, prod_url)
    try_time += 1
    if try_time > 10:
        raise Exception("[INFO] Fail to push data to ESB!!!")
print(try_time)
print("[INFO] Push finished! ( ^ _ ^ ) V")

# In[ ]:
