# _*_ coding: utf-8 _*_

"""
Prepare level-3 distribution data.

Author: Genpeng Xu
"""

import os
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.preprocessing import LabelEncoder

# Own Customized modules
from base.base_data_loader import BaseDataLoader
from util.data_util import transform_channel, remove_whitespace
from util.date_util import get_days_of_month, infer_month
from util.feature_util import (prepare_training_set_for_level3,
                               prepare_val_set_for_level3,
                               prepare_testing_set_for_level3,
                               modify_training_set)
from global_vars import (DIS_DATA_DIR, DIS_DATA_COLUMN_NAMES,
                         SUPPORTED_CATE_NAMES, CATE_NAME_2_CATE_CODE)


class Level3DisDataLoader(BaseDataLoader):
    """Distribution data loader of Level-3 (per customer per sku)."""

    def __init__(self, year, month, categories='all', need_unitize=True):
        self._year, self._month = year, month
        self._categories = categories
        self._all_cate_names = self._get_all_cates(self._categories)
        self._all_cate_codes = set([CATE_NAME_2_CATE_CODE[cate] for cate in self._all_cate_names])
        self._version_flag = "%d-%02d" % (self._year, self._month)
        self._dis = self._get_dis_data(need_unitize)
        self._dis_cus_sku_month = self._get_month_dis_per_cus_sku()  # 得到每个代理商每个SKU每个月分销
        self._index = self._dis_cus_sku_month.index
        self._dis_cus_sku_month_pre15 = self._get_pre15_dis_per_cus_sku()  # 得到每个代理商每个SKU前15天的分销
        self._customer_info, self._customer_info_encoded = self._get_cus_info()  # 得到代理商的信息
        self._sku_info, self._sku_info_encoded = self._get_sku_info()  # 得到SKU的信息
        self._dis_sku_month = self._get_month_dis_per_sku()  # 得到每个SKU每个月的分销
        self._dis_cate1_month = self._get_month_dis_per_cate1()  # 得到每个大类每个月的分销
        self._dis_cate2_month = self._get_month_dis_per_cate2()  # 得到每个小类每个月的分销
        self._dis_cus_cate1_month = self._get_month_dis_per_cus_cate1()  # 得到每个代理商每个大类的分销
        self._dis_cus_cate2_month = self._get_month_dis_per_cus_cate2()  # 得到每个代理商每个小类的分销
        self._dis_cus_chan_month = self._get_month_dis_per_cus_chan()  # 得到每个代理商每个渠道的分销
        self._dis_cus_sales_chan_month = self._get_month_dis_per_cus_sales_chan()  # 得到每个代理商每个销售渠道的分销

    def _get_all_cates(self, categories):
        if isinstance(categories, list):
            all_cates = set([cate for cate in categories if cate in SUPPORTED_CATE_NAMES])
        elif categories == 'all':
            all_cates = set(SUPPORTED_CATE_NAMES)
        else:
            raise Exception("[INFO] The input `categories` is illegal!!!")
        return all_cates

    def _get_data_path(self):
        filename = "m111-dis_%s.txt" % self._version_flag
        return os.path.join(DIS_DATA_DIR, self._version_flag, filename)

    def _get_dis_data(self, need_unitize=True):
        print("[INFO] Start loading distribution data...")
        dis_data_path = self._get_data_path()
        dis = pd.read_csv(
            dis_data_path, sep='\u001e', header=None,
            names=DIS_DATA_COLUMN_NAMES, parse_dates=[0],
            dtype={5: str, 7: str, 15: str, 22: str, 27: str, 28: str}
        )
        print("[INFO] Loading finished!")
        print("[INFO] Start preprocessing distribution data...")
        dis = self._preprocess_dis_data(dis, need_unitize)
        print("[INFO] Preprocessing finished!")
        return dis

    def _preprocess_dis_data(self, dis, need_unitize=True):
        dis.drop(
            columns=['bu_code', 'bu_name', 'region_code', 'region_name', 'road'],
            inplace=True
        )
        dis = dis.loc[dis.first_cate_code.isin(self._all_cate_codes)]
        dis = dis.loc[dis.customer_code.str.startswith('C')]
        dis = dis.sort_values(by='order_date').reset_index(drop=True)
        str_column_names = ['sales_cen_code', 'customer_code', 'item_code']
        remove_whitespace(dis, str_column_names)
        dis['district'] = dis.district.str.replace(r'\\N', '未知')
        dis['customer_type'] = dis.customer_type.str.replace(r'\\N', '未知')
        dis['is_usable'] = dis.is_usable.astype(str)
        dis['is_usable'] = dis.is_usable.str.replace(r'\\N', '未知')
        dis['channel_name'] = dis.channel_name.apply(lambda x: transform_channel(x))
        dis['dis_qty'] = np.round(dis.dis_qty).astype(int)
        dis['item_price'] = dis.item_price.astype(str).str.replace(r'\\N', '-1.0').astype(float)
        dis['dis_amount'] = dis.dis_qty * dis.item_price
        if need_unitize:
            dis['dis_qty'] = dis.dis_qty / 10000
            dis['dis_amount'] = dis.dis_amount / 10000
        return dis

    def _get_month_dis_per_cus_sku(self):
        """Get monthly distribution data per customer per sku."""
        tmp = self._dis.copy()
        tmp['order_month'] = tmp.order_date.astype(str).apply(lambda x: x[:7])
        tmp = tmp.groupby(['customer_code', 'item_code', 'order_month'])[['dis_qty']].sum()
        tmp['dis_qty'] = tmp.dis_qty.apply(lambda x: x if x > 0 else 0)
        dis_cus_sku_month = tmp.unstack(level=-1).fillna(0)
        dis_cus_sku_month.columns = pd.date_range(
            start='2018-01-30', periods=len(dis_cus_sku_month.columns), freq='M'
        )
        return dis_cus_sku_month

    def _get_pre15_dis_per_cus_sku(self):
        """Get half monthly distribution data per customer per sku."""
        tmp = self._dis.copy()
        tmp['day'] = tmp.order_date.dt.day
        tmp = tmp.loc[tmp.day <= 15]
        tmp['order_month'] = tmp.order_date.astype(str).apply(lambda x: x[:7])
        dis_cus_sku_month_pre15 = tmp.groupby(['customer_code', 'item_code', 'order_month'])[['dis_qty']].sum()
        dis_cus_sku_month_pre15['dis_qty'] = dis_cus_sku_month_pre15.dis_qty.apply(lambda x: 0 if x < 0 else x)
        dis_cus_sku_month_pre15 = dis_cus_sku_month_pre15.unstack(level=-1).fillna(0.0)
        dis_cus_sku_month_pre15.columns = pd.date_range(
            start='2018-01-30', periods=len(dis_cus_sku_month_pre15.columns), freq='M'
        )
        dis_cus_sku_month_pre15 = dis_cus_sku_month_pre15.reindex(self._index).fillna(0)
        return dis_cus_sku_month_pre15

    def _get_month_dis_per_sku(self):
        """Get monthly distribution data per sku."""
        dis_sku_month = self._dis_cus_sku_month.groupby(['item_code'])[self._dis_cus_sku_month.columns].sum()
        dis_sku_month = dis_sku_month.reindex(self._index.get_level_values(1))
        return dis_sku_month

    def _get_cus_info(self):
        """Get information of all customers."""
        label_enc = LabelEncoder()
        customer_info = self._dis.drop_duplicates(['customer_code'], keep='last')
        customer_info = customer_info[['customer_code', 'customer_name', 'sales_cen_code',
                                       'sales_cen_name', 'sales_region_name', 'province',
                                       'city', 'district', 'customer_type', 'is_usable', 'channel_level']]
        customer_info['customer_id'] = label_enc.fit_transform(customer_info['customer_code'])
        customer_info['sales_cen_id'] = label_enc.fit_transform(customer_info['sales_cen_code'])
        customer_info['sales_region_id'] = label_enc.fit_transform(customer_info['sales_region_name'])
        customer_info['province_id'] = label_enc.fit_transform(customer_info['province'])
        customer_info['city_id'] = label_enc.fit_transform(customer_info['city'])
        customer_info['district_id'] = label_enc.fit_transform(customer_info['district'])
        customer_info['customer_type'] = label_enc.fit_transform(customer_info['customer_type'])
        customer_info['is_usable'] = label_enc.fit_transform(customer_info['is_usable'])
        customer_info['channel_level'] = label_enc.fit_transform(customer_info['channel_level'])
        customer_info_encoded = customer_info.drop(
            columns=['customer_name', 'sales_cen_code', 'sales_cen_name',
                     'sales_region_name', 'province', 'city', 'district']
        ).set_index('customer_code')
        customer_info.set_index('customer_code', inplace=True)
        customer_info_encoded = customer_info_encoded.reindex(self._index.get_level_values(0))
        return customer_info, customer_info_encoded

    def _get_sku_info(self):
        """Get information of all SKUs."""
        label_enc = LabelEncoder()
        sku_info = self._dis.drop_duplicates(['item_code'], keep='last')
        sku_info = sku_info[[
            'item_code', 'item_name', 'first_cate_code',
            'first_cate_name', 'second_cate_code', 'second_cate_name',
            'item_price', 'channel_name', 'sales_chan_name'
        ]]
        sku_info['item_id'] = label_enc.fit_transform(sku_info.item_code)
        sku_info['first_cate_id'] = label_enc.fit_transform(sku_info.first_cate_code)
        sku_info['second_cate_id'] = label_enc.fit_transform(sku_info.second_cate_code)
        sku_info['channel_id'] = label_enc.fit_transform(sku_info.channel_name)
        sku_info['sales_chan_id'] = label_enc.fit_transform(sku_info.sales_chan_name)
        sku_info_encoded = sku_info.drop(
            columns=['item_name', 'first_cate_code', 'first_cate_name',
                     'second_cate_code', 'second_cate_name', 'channel_name', 'sales_chan_name']
        ).set_index('item_code')
        sku_info = sku_info.set_index('item_code')
        sku_info_encoded = sku_info_encoded.reindex(self._index.get_level_values(1))
        return sku_info, sku_info_encoded

    def _get_month_dis_per_cate1(self):
        """Get monthly distribution data per first level category."""
        dis_cate1_month = self._dis_cus_sku_month.reset_index()
        dis_cate1_month['first_cate_id'] = self._sku_info_encoded.first_cate_id.values
        dis_cate1_month_index = dis_cate1_month['first_cate_id']
        dis_cate1_month = dis_cate1_month.groupby(['first_cate_id'])[self._dis_cus_sku_month.columns].sum()
        dis_cate1_month = dis_cate1_month.reindex(dis_cate1_month_index)
        return dis_cate1_month

    def _get_month_dis_per_cate2(self):
        """Get monthly distribution data per second level category."""
        dis_cate2_month = self._dis_cus_sku_month.reset_index()
        dis_cate2_month['second_cate_id'] = self._sku_info_encoded.second_cate_id.values
        dis_cate2_month_index = dis_cate2_month['second_cate_id']
        dis_cate2_month = dis_cate2_month.groupby(['second_cate_id'])[self._dis_cus_sku_month.columns].sum()
        dis_cate2_month = dis_cate2_month.reindex(dis_cate2_month_index)
        return dis_cate2_month

    def _get_month_dis_per_cus_cate1(self):
        """Get monthly distribution data per customer per first level category."""
        dis_cus_cate1_month = self._dis_cus_sku_month.reset_index()
        dis_cus_cate1_month['first_cate_id'] = self._sku_info_encoded.first_cate_id.values
        dis_cus_cate1_month_index = dis_cus_cate1_month[['customer_code', 'first_cate_id']]
        dis_cus_cate1_month = dis_cus_cate1_month.groupby(
            ['customer_code', 'first_cate_id']
        )[self._dis_cus_sku_month.columns].sum()
        dis_cus_cate1_month = dis_cus_cate1_month.reindex(dis_cus_cate1_month_index)
        return dis_cus_cate1_month

    def _get_month_dis_per_cus_cate2(self):
        """Get monthly distribution data per customer per second level category."""
        dis_cus_cate2_month = self._dis_cus_sku_month.reset_index()
        dis_cus_cate2_month['second_cate_id'] = self._sku_info_encoded.second_cate_id.values
        dis_cus_cate2_month_index = dis_cus_cate2_month[['customer_code', 'second_cate_id']]
        dis_cus_cate2_month = dis_cus_cate2_month.groupby(
            ['customer_code', 'second_cate_id']
        )[self._dis_cus_sku_month.columns].sum()
        dis_cus_cate2_month = dis_cus_cate2_month.reindex(dis_cus_cate2_month_index)
        return dis_cus_cate2_month

    def _get_month_dis_per_cus_chan(self):
        """Get monthly distribution data per customer per channel."""
        dis_cus_chan_month = self._dis_cus_sku_month.reset_index()
        dis_cus_chan_month['channel_id'] = self._sku_info_encoded.channel_id.values
        dis_cus_chan_month_index = dis_cus_chan_month[['customer_code', 'channel_id']]
        dis_cus_chan_month = dis_cus_chan_month.groupby(
            ['customer_code', 'channel_id']
        )[self._dis_cus_sku_month.columns].sum()
        dis_cus_chan_month = dis_cus_chan_month.reindex(dis_cus_chan_month_index)
        return dis_cus_chan_month

    def _get_month_dis_per_cus_sales_chan(self):
        """Get monthly distribution data per customer per sales channel."""
        dis_cus_sales_chan_month = self._dis_cus_sku_month.reset_index()
        dis_cus_sales_chan_month['sales_chan_id'] = self._sku_info_encoded.sales_chan_id.values
        dis_cus_sales_chan_month_index = dis_cus_sales_chan_month[['customer_code', 'sales_chan_id']]
        dis_cus_sales_chan_month = dis_cus_sales_chan_month.groupby(
            ['customer_code', 'sales_chan_id'])[self._dis_cus_sku_month.columns].sum()
        dis_cus_sales_chan_month = dis_cus_sales_chan_month.reindex(dis_cus_sales_chan_month_index)
        return dis_cus_sales_chan_month

    def prepare_training_set(self, months, gap=0):
        X_train, y_train = prepare_training_set_for_level3(None, self._dis_cus_sku_month, None,
                                                           None, self._dis_cus_sku_month_pre15, None,
                                                           None, self._dis_cus_cate1_month, None,
                                                           None, self._dis_cus_cate2_month, None,
                                                           None, self._dis_cus_chan_month, None,
                                                           None, self._dis_cus_sales_chan_month, None,
                                                           None, self._dis_sku_month, None,
                                                           None, self._dis_cate1_month, None,
                                                           None, self._dis_cate2_month, None,
                                                           self._customer_info_encoded, self._sku_info_encoded,
                                                           months, gap, label_data='dis')
        return modify_training_set(X_train, y_train)

    def prepare_val_set(self, pred_year, pred_month, gap=0):
        return prepare_val_set_for_level3(None, self._dis_cus_sku_month, None,
                                          None, self._dis_cus_sku_month_pre15, None,
                                          None, self._dis_cus_cate1_month, None,
                                          None, self._dis_cus_cate2_month, None,
                                          None, self._dis_cus_chan_month, None,
                                          None, self._dis_cus_sales_chan_month, None,
                                          None, self._dis_sku_month, None,
                                          None, self._dis_cate1_month, None,
                                          None, self._dis_cate2_month, None,
                                          self._customer_info_encoded, self._sku_info_encoded,
                                          pred_year, pred_month, gap, label_data='dis')

    def prepare_testing_set(self, pred_year, pred_month, gap=0):
        return prepare_testing_set_for_level3(None, self._dis_cus_sku_month, None,
                                              None, self._dis_cus_sku_month_pre15, None,
                                              None, self._dis_cus_cate1_month, None,
                                              None, self._dis_cus_cate2_month, None,
                                              None, self._dis_cus_chan_month, None,
                                              None, self._dis_cus_sales_chan_month, None,
                                              None, self._dis_sku_month, None,
                                              None, self._dis_cate1_month, None,
                                              None, self._dis_cate2_month, None,
                                              self._customer_info_encoded, self._sku_info_encoded,
                                              pred_year, pred_month, gap)

    def get_true_data(self, true_pred_year: int, true_pred_month: int, reset_index: bool = False) -> pd.DataFrame:
        start_dt_str = "%d-%02d-01" % (true_pred_year, true_pred_month)
        end_dt_str = "%d-%02d-%02d" % (true_pred_year,
                                       true_pred_month,
                                       get_days_of_month(true_pred_year, true_pred_month))
        df = self._dis.loc[(self._dis.order_date >= start_dt_str) & (self._dis.order_date <= end_dt_str)]
        df['order_date'] = df.order_date.astype(str).apply(lambda x: x[:4] + x[5:7])
        df = df.groupby(['customer_code', 'item_code', 'order_date'])[['dis_qty']].sum()
        df = df.loc[df.dis_qty > 0]
        df.rename(columns={'inv_qty': 'act_inv_qty'}, inplace=True)
        return df.reset_index() if reset_index else df

    def add_index(self,
                  preds: Union[np.ndarray, List[np.ndarray]],
                  start_pred_year: int,
                  start_pred_month: int) -> pd.DataFrame:
        if isinstance(preds, np.ndarray):
            preds = [preds]
        months_pred = ['%d%02d' % infer_month(start_pred_year, start_pred_month, i) for i in range(len(preds))]
        return pd.DataFrame(np.array(preds).transpose(), index=self._index, columns=months_pred)

    def decorate_pred_result(self,
                             preds: Union[np.ndarray, List[np.ndarray]],
                             start_pred_year: int,
                             start_pred_month: int,
                             use_unitize: bool = True) -> pd.DataFrame:
        df_preds = self.add_index(preds, start_pred_year, start_pred_month).stack().to_frame('pred_dis_qty')
        df_preds.index.set_names(['customer_code', 'item_code', 'order_date'], inplace=True)
        df_preds['pred_dis_qty'] = df_preds.pred_dis_qty.apply(lambda x: x if x > 0 else 0)
        df_preds['pred_dis_qty'] = np.round(df_preds.pred_dis_qty, decimals=4 if use_unitize else 0)
        return df_preds

    def predict_by_history(self, start_pred_year, start_pred_month, gap=4, left_bound_dt='2018-01'):
        left_bound_year, left_bound_month = map(int, left_bound_dt.split('-'))
        start_aver_year, start_aver_month = infer_month(start_pred_year, start_pred_month, gap)
        pred_len = 12 - gap
        history = []
        for i in range(1, 4):
            if (start_aver_year - i > left_bound_year) or \
                    (start_aver_year - i == left_bound_year and start_aver_month >= left_bound_month):
                start_dt = "%d-%02d-%d" % (start_aver_year - i, start_aver_month, 1)
                tmp = self._dis_cus_sku_month[pd.date_range(start_dt, periods=pred_len, freq='M')].values
                history.append(tmp)
        result = np.mean(np.array(history), axis=0)
        months_pred = ['%d%02d' % infer_month(start_aver_year, start_aver_month, i) for i in range(pred_len)]
        return pd.DataFrame(result, index=self._index, columns=months_pred)

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def all_cate_name(self):
        return self._all_cate_names

    @property
    def version_flag(self):
        return self._version_flag

    @property
    def customer_info(self):
        return self._customer_info

    @property
    def sku_info(self):
        return self._sku_info


if __name__ == '__main__':
    curr_year, curr_month = 2019, 12
    level3_dis_data = Level3DisDataLoader(curr_year, curr_month)
    X_val, y_val = level3_dis_data.prepare_val_set(2019, 9)
    print(X_val.shape)
    print(y_val.shape)
