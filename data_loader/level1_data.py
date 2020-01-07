# _*_ coding: utf-8 _*_

"""
Prepare level-3 order data.

Author: Genpeng Xu
"""

import os
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.preprocessing import LabelEncoder

# Own customized variables & modules
from base.base_data_loader import BaseDataLoader
from util.date_util import get_days_of_month, infer_month
from util.data_util import transform_channel, transform_project_flag, remove_whitespace
from util.feature_util import (prepare_training_set_for_level1,
                               prepare_val_set_for_level1,
                               prepare_testing_set_for_level1,
                               modify_training_set,
                               get_pre_vals,
                               get_val)
from global_vars import (ORDER_DATA_DIR, ORDER_DATA_COLUMN_NAMES,
                         SUPPORTED_CATE_NAMES, CATE_NAME_2_CATE_CODE,
                         DIS_DATA_DIR, DIS_DATA_COLUMN_NAMES,
                         INV_DATA_DIR, INV_DATA_COLUMN_NAMES)


class Level1DataLoader(BaseDataLoader):
    """Data loader of Level-2 (per sku)."""

    def __init__(self, year, month, categories='all', need_unitize=True, label_data='order'):
        self._year, self._month = year, month
        self._categories = categories
        self._label_data = label_data
        self._all_cate_names = self._get_all_categories(self._categories)  # 品类名
        self._all_cate_codes = set([CATE_NAME_2_CATE_CODE[cn] for cn in self._all_cate_names])
        self._version_flag = "%d-%02d" % (self._year, self._month)
        self._order = self._get_order_data(need_unitize)  # 提货数据
        self._order_cate1_month = self._get_month_order_per_cate1()  # 得到每个大类每个月的提货
        self._index = self._get_index(label_data)
        self._reset_index()
        self._cate_info = self._get_cate_info()

    def _get_all_categories(self, categories):
        if isinstance(categories, list):
            all_cates = set([cate for cate in categories if cate in SUPPORTED_CATE_NAMES])
        elif categories == 'all':
            all_cates = set(SUPPORTED_CATE_NAMES)
        else:
            raise Exception("[INFO] The input `categories` is illegal!!!")
        return all_cates

    def _get_order_data_filepath(self):
        filename = "m111-order_%s.txt" % self._version_flag
        return os.path.join(ORDER_DATA_DIR, self._version_flag, filename)

    def _get_order_data(self, need_unitize=True):
        print("[INFO] Start loading order data...")
        order_data_path = self._get_order_data_filepath()
        order = pd.read_csv(
            order_data_path, sep='\u001e', header=None, names=ORDER_DATA_COLUMN_NAMES,
            parse_dates=[0], dtype={14: str, 20: str, 22: str, 30: str}
        )
        print("[INFO] Loading finished!")
        print("[INFO] Start preprocessing order data...")
        order = self._preprocess_order_data(order, need_unitize)
        print("[INFO] Preprocessing finished!")
        return order

    def _preprocess_order_data(self, order, need_unitize=True):
        order = order.loc[order.first_cate_code.isin(self._all_cate_codes)]
        order = order.loc[order.order_date >= '2015-09-01']
        order = order.loc[~order.customer_code.str.contains('\\N', regex=False)]
        removed_column_names = [
            'org_code', 'region_code', 'region_name', 'road',
            'fin_segment1_code', 'fin_segment1_name',
            'fin_segment2_code', 'fin_segment2_name',
            'received_qty', 'return_qty']
        order.drop(columns=removed_column_names, inplace=True)
        order = order.sort_values(by='order_date').reset_index(drop=True)
        str_column_names = ['sales_cen_code', 'customer_code', 'item_code']
        remove_whitespace(order, str_column_names)
        order['district'] = order.district.str.replace(r'\\N', '未知')
        order['channel_name'] = order.channel_name.apply(lambda x: transform_channel(x))
        order['channel_name'] = order.channel_name.str.replace(r'\\N', '未知')
        order['sales_chan_name'] = order.sales_chan_name.str.replace(r'\\N', '未知')
        order.project_flag.fillna('未知', inplace=True)
        order['project_flag'] = order.project_flag.apply(lambda x: transform_project_flag(x))
        if need_unitize:  # 是否需要单位化，即以万作为单位
            order['ord_qty'] = order.ord_qty / 10000
            order['ord_amount'] = order.ord_amount / 10000
        return order

    def _get_month_order_per_cate1(self):
        """Get monthly order data per first level category."""
        order_cate1_month = self._order.copy()
        order_cate1_month['order_month'] = order_cate1_month.order_date.astype('str').apply(lambda x: x[:7])
        order_cate1_month = order_cate1_month.groupby(['first_cate_code', 'order_month'])[['ord_qty']].sum()
        order_cate1_month = order_cate1_month.unstack(level=-1).fillna(0.0)
        order_cate1_month.columns = pd.date_range(start='2015-09-30', periods=len(order_cate1_month.columns), freq='M')
        return order_cate1_month

    def _get_cate_info(self):
        label_enc = LabelEncoder()
        cate_info = self._order.copy()
        cate_info = cate_info.loc[cate_info.item_price > 10]
        cate_info = cate_info.groupby(['first_cate_code'])[['ord_qty', 'ord_amount']].sum()
        cate_info['cate_aver_price'] = cate_info.ord_amount // cate_info.ord_qty
        cate_info = cate_info.drop(columns=['ord_qty', 'ord_amount'])
        cate_info = cate_info.reindex(self._index).reset_index()
        cate_info['first_cate_id'] = label_enc.fit_transform(cate_info.first_cate_code)
        cate_info = cate_info.set_index('first_cate_code')
        return cate_info

    def _get_dis_data_filepath(self):
        filepath = "m111-dis_%s.txt" % self._version_flag
        return os.path.join(DIS_DATA_DIR, self._version_flag, filepath)

    def _get_dis_data(self, need_unitize=True):
        print("[INFO] Start loading distribution data...")
        filepath = self._get_dis_data_filepath()
        dis = pd.read_csv(
            filepath, sep='\u001e', header=None, names=DIS_DATA_COLUMN_NAMES,
            parse_dates=[0], dtype={5: str, 7: str, 15: str, 22: str, 27: str, 28: str}
        )
        print("[INFO] Loading finished!")
        print("[INFO] Start preprocessing distribution data...")
        dis = self._preprocess_dis_data(dis, need_unitize)
        print("[INFO] Preprocessing finished!")
        return dis

    def _preprocess_dis_data(self, dis, need_unitize=True):
        dis.drop(columns=['bu_code',
                          'bu_name',
                          'region_code',
                          'region_name',
                          'road'], inplace=True)
        dis = dis.loc[dis.first_cate_code.isin(self._all_cate_codes)]
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
        if need_unitize:
            dis['dis_qty'] = dis.dis_qty / 10000
            dis['dis_amount'] = dis.dis_amount / 10000
        return dis

    def _get_inv_data_filepath(self):
        filename = "m111-inv_%s.txt" % self._version_flag
        return os.path.join(INV_DATA_DIR, self._version_flag, filename)

    def _get_inv_data(self, need_unitize=True):
        print("[INFO] Start loading inventory data...")
        filepath = self._get_inv_data_filepath()
        inv = pd.read_csv(
            filepath, sep='\u001e', header=None, names=INV_DATA_COLUMN_NAMES,
            parse_dates=[0], dtype={3: str, 5: str}
        )
        print("[INFO] Loading finished!")
        print("[INFO] Start preprocessing inventory data...")
        inv = self._preprocess_inv_data(inv, need_unitize)
        print("[INFO] Preprocessing finished!")
        return inv

    def _preprocess_inv_data(self, inv, need_unitize=True):
        inv.drop(columns=['bu_code',
                          'bu_name',
                          'region_code',
                          'region_name'], inplace=True)
        inv = inv.loc[inv.first_cate_code.isin(self._all_cate_codes)]
        inv = inv.sort_values(by='order_date').reset_index(drop=True)
        inv['sales_cen_code'] = inv.sales_cen_code.astype(str).apply(lambda x: x.strip())
        inv['customer_code'] = inv.customer_code.astype(str).apply(lambda x: x.strip())
        inv['item_code'] = inv.item_code.astype(str).apply(lambda x: x.strip())
        inv['district'] = inv.district.str.replace(r'\\N', '未知')
        inv['channel_name'] = inv.channel_name.apply(lambda x: transform_channel(x))
        inv['sales_chan_name'] = inv.sales_chan_name.str.replace(r'\\N', '未知')
        if need_unitize:
            inv['inv_qty'] = inv.inv_qty / 10000
            inv['inv_amount'] = inv.inv_amount / 10000
        return inv

    def _get_index(self, label_data):
        if label_data == 'order':
            return self._order_cate1_month.index
        elif label_data == 'dis':
            return self._dis_cate1_month.index
        elif label_data == 'inv':
            return self._inv_cate1_month.index
        else:
            raise Exception("[INFO] The type of label data is illegal! " +
                            "Must be one of the three types 'order', 'dis', and 'inv'.")

    def _reset_index(self) -> None:
        self._order_cate1_month = self._order_cate1_month.reindex(self._index)
        # self._dis_cate1_month = self._dis_cate1_month.reindex(self._index)
        # self._inv_cate1_month = self._inv_cate1_month.reindex(self._index)

    def prepare_training_set(self, months, gap=0):
        X_train, y_train = prepare_training_set_for_level1(self._order_cate1_month, None, None,
                                                           self._cate_info, months, gap, label_data=self._label_data)
        return modify_training_set(X_train, y_train)

    def prepare_val_set(self, pred_year, pred_month, gap=0):
        return prepare_val_set_for_level1(self._order_cate1_month, None, None,
                                          self._cate_info, pred_year, pred_month, gap, label_data=self._label_data)

    def prepare_testing_set(self, pred_year, pred_month, gap=0):
        return prepare_testing_set_for_level1(self._order_cate1_month, None, None,
                                              self._cate_info, pred_year, pred_month, gap, label_data=self._label_data)

    def get_true_order_data(self,
                            true_pred_year: int,
                            true_pred_month: int,
                            reset_index: bool = False) -> pd.DataFrame:
        start_dt_str = "%d-%02d-01" % (true_pred_year, true_pred_month)
        end_dt_str = "%d-%02d-%02d" % (true_pred_year,
                                       true_pred_month,
                                       get_days_of_month(true_pred_year, true_pred_month))
        df_act = self._order.loc[(self._order.order_date >= start_dt_str) & (self._order.order_date <= end_dt_str)]
        df_act['order_date'] = df_act.order_date.astype(str).apply(lambda x: x[:4] + x[5:7])
        df_act = df_act.groupby(['first_cate_code', 'order_date'])[['ord_qty']].sum()
        df_act = df_act.loc[df_act.ord_qty > 0]
        df_act.rename(columns={'ord_qty': 'act_ord_qty'}, inplace=True)
        return df_act.reset_index() if reset_index else df_act

    def add_index(self,
                  preds: Union[np.ndarray, List[np.ndarray]],
                  start_pred_year: int,
                  start_pred_month: int) -> pd.DataFrame:
        if isinstance(preds, np.ndarray):
            preds = [preds]
        months_pred = ['%d%02d' % infer_month(start_pred_year, start_pred_month, i) for i in range(len(preds))]
        return pd.DataFrame(np.array(preds).transpose(), index=self._index, columns=months_pred)

    def add_index_v2(self, preds: Union[np.ndarray, List[np.ndarray]]) -> pd.DataFrame:
        if isinstance(preds, np.ndarray):
            preds = [preds]
        months_pred = ['pred_ord_qty_m%d' % i for i in range(1, len(preds) + 1)]
        return pd.DataFrame(np.array(preds).transpose(), index=self._index, columns=months_pred)

    def decorate_pred_result(self,
                             preds: Union[np.ndarray, List[np.ndarray]],
                             start_pred_year: int,
                             start_pred_month: int,
                             use_unitize: bool = True) -> pd.DataFrame:
        df_preds = self.add_index(preds, start_pred_year, start_pred_month).stack().to_frame('pred_ord_qty')
        df_preds.index.set_names(['first_cate_code', 'order_date'], inplace=True)
        df_preds['pred_ord_qty'] = df_preds.pred_ord_qty.apply(lambda x: x if x > 0 else 0)
        df_preds['pred_ord_qty'] = np.round(df_preds.pred_ord_qty, decimals=4 if use_unitize else 0)
        return df_preds

    def predict_by_history(self, start_pred_year, start_pred_month, gap=4, left_bound_dt='2015-09'):
        left_bound_year, left_bound_month = map(int, left_bound_dt.split('-'))
        start_aver_year, start_aver_month = infer_month(start_pred_year, start_pred_month, gap)
        pred_len = 12 - gap
        history = []
        for i in range(1, 4):
            if (start_aver_year - i > left_bound_year) or \
                    (start_aver_year - i == left_bound_year and start_aver_month >= left_bound_month):
                start_dt = "%d-%02d-%d" % (start_aver_year - i, start_aver_month, 1)
                tmp = self._order_sku_month[pd.date_range(start_dt, periods=pred_len, freq='M')].values
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
    def all_cate_names(self):
        return self._all_cate_names

    @property
    def version_flag(self):
        return self._version_flag

    @property
    def cate_info(self):
        return self._cate_info


if __name__ == '__main__':
    curr_year, curr_month, _ = 2019, 12, 10
    level1_order_data = Level1DataLoader(curr_year, curr_month)
    X_val, y_val = level1_order_data.prepare_val_set(2019, 9)
    print(X_val.shape)
    print(y_val.shape)
