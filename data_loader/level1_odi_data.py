# _*_ coding: utf-8 _*_

"""
Prepare level1 data, including order, distribution and inventory.

Author: Genpeng Xu
"""

# Third-party libraries
import numpy as np
import pandas as pd
# from category_encoders import BinaryEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Own scaffolds
from util.date_util import *
from util.feature_util import prepare_training_set_for_level3, prepare_testing_set
from base.base_loader import BaseLoader


class Level1OdiData(BaseLoader):
    """Order, distribution and inventory data."""

    def __init__(self, start_pred_year, start_pred_month, start_pred_day, categories):
        self._start_pred_year = start_pred_year
        self._start_pred_month = start_pred_month
        self._start_pred_day = start_pred_day
        self._all_categories = categories
        self._order = self._load_order_data(categories)
        self._level1_order = self._prepare_level1_order()
        self._level1_dis = None
        self._level1_inv = None
        self._category_info = self._prepare_category_info(categories)
        self._cate_aver_price = self._calc_cate_aver_price()

    def _load_order_data(self, categories):
        """
        Load order data.
        TODO: modify to load from database

        Arguments:
            ord_path : str, the path of order data
            categories : list of str, all the categories we want to predict

        Return:
             order data.
        """
        ord_path = "data/level2/m111-sku-order-all-final.csv"
        order = pd.read_csv(
            ord_path, sep=',', parse_dates=['order_date']
        ).rename(columns={'sales_class_1': 'category'})
        order = order.loc[order.category.isin(categories)]
        end_dt_str = "%d-%02d-%02d" % (self._start_pred_year, self._start_pred_month, self._start_pred_day)
        order = order.loc[(order.order_date >= '2015-09-01') & (order.order_date <= end_dt_str)]
        order = order.sort_values(by='order_date').reset_index(drop=True)
        order['ord_qty'] = order.ord_qty / 10000
        order['ord_amount'] = order.ord_amount / 10000
        return order

    def _prepare_level1_order(self):
        """
        Prepare level1 order data whose granularity is category.

        Return:
            level1 order data.
        """
        end_dt_str = '%d-%02d-%d' % (self._start_pred_year,
                                     self._start_pred_month,
                                     get_days_of_month(self._start_pred_year, self._start_pred_month))
        # order amount per item per month
        order_cate_month = self._order.copy()
        order_cate_month['order_month'] = order_cate_month.order_date.astype('str').apply(lambda x: x[:7])
        order_cate_month = order_cate_month.groupby(['category', 'order_month'])[['ord_qty']].sum()
        order_cate_month = order_cate_month.unstack(level=-1).fillna(0.0)
        order_cate_month.columns = pd.date_range(start='2015-09-30', end=end_dt_str, freq='M')
        return order_cate_month

    def _prepare_category_info(self, categories, encode_method='ordinal'):
        """
        Encode the category code, and return.

        Return:
            category information.
        """
        category = pd.DataFrame(categories, columns=['category'])
        if encode_method == 'binary':
            # binary encoding
            binary_enc = BinaryEncoder(cols=['category'])
            category_binary = binary_enc.fit_transform(category)
            category = pd.concat([category, category_binary], axis=1)
            category.set_index('category', inplace=True)
        elif encode_method == 'onehot':
            # onehot encoding
            onehot_enc = OneHotEncoder(cols=['category'])
            category_onehot = onehot_enc.fit_transform(category)
            category = pd.concat([category, category_onehot], axis=1)
            category.set_index('category', inplace=True)
        else:
            # ordinal encoding (Default)
            label_enc = LabelEncoder()
            category['category_id'] = label_enc.fit_transform(category.category)
            category.set_index('category', inplace=True)
        category = category.reindex(self._level1_order.index)
        return category

    def _calc_cate_aver_price(self):
        """
        Calculate average price of each category.

        Return:
            average price of each category
        """
        order = self._order.copy()
        cate_aver_price = order.groupby(['category'])[['price']].mean()
        cate_aver_price['price'] = np.round(cate_aver_price.price, decimals=2)
        # cate_aver_price['price'] = cate_aver_price.price.astype(np.int32)
        cate_aver_price = cate_aver_price.reindex(self._level1_order.index)
        return cate_aver_price

    def prepare_data_for_predict(self, periods=4, left_bound='2016-01'):
        """
        Prepare data for predicting.

        Arguments:
            periods : int, the length to predict
            left_bound : str (format: 'year-month'), first date to extract features

        Return:
            X_train : the features of training data
            y_train : the labels of training data
            X_test : the features of test data
        """
        # prepare training data
        year_upper_bound, month_upper_bound = infer_month(self._start_pred_year,
                                                          self._start_pred_month,
                                                          offset=-(periods+1))
        train_pairs = get_pre_months(year_upper_bound,
                                     month_upper_bound,
                                     left_bound=left_bound)
        X_l, y_l = [], []
        for pair in train_pairs:
            y, m = int(pair.split('-')[0]), int(pair.split('-')[1])
            # 有添加M月前10天提货、分销、库存量
            # pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, y, m)
            # X_tmp, y_tmp = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, y, m)
            # X_tmp = pd.concat([X_tmp, pre_10_days, cates.reset_index(drop=True)], axis=1)

            # 没有添加M月前10天提货、分销、库存量
            X_tmp, y_tmp = prepare_dataset(self._level1_order, None, None, y, m, periods=periods)
            X_tmp = pd.concat([X_tmp,
                               self._category_info.reset_index(drop=True),
                               self._cate_aver_price.reset_index(drop=True)], axis=1)
            X_l.append(X_tmp)
            y_l.append(y_tmp)
        X_train = pd.concat(X_l, axis=0)
        y_train = np.concatenate(y_l, axis=0)

        # prepare test data
        # pre_10_days = get_pre_10_days(order, dis, inv, order_cate_month.index, 2018, 11)
        # X_test = prepare_dataset(order_cate_month, dis_cate_month, inv_cate_month, 2018, 11, is_train=False)
        # X_test = pd.concat([X_test, pre_10_days, cates.reset_index(drop=True)], axis=1)
        X_test = prepare_dataset(self._level1_order,
                                 None,
                                 None,
                                 self._start_pred_year,
                                 self._start_pred_month,
                                 periods=periods,
                                 is_train=False)
        X_test = pd.concat([X_test,
                            self._category_info.reset_index(drop=True),
                            self._cate_aver_price.reset_index(drop=True)], axis=1)
        return X_train, y_train, X_test

    def get_cate_order(self, start_dt_str, end_dt_str):
        """
        Calculate the order quantity in a certain period.

        Arguments:
            start_dt_str : str (format: year-month-day), the start date
            end_dt_str : str (format: year-month-day), the end date
        
        Return:
            ret : DataFrame, the order quantity in a certain period
        """
        order_m = self._order.loc[(self._order.order_date >= start_dt_str) & (self._order.order_date <= end_dt_str)]
        order_m['order_date'] = '%d-%02d' % (start_pred_year, start_pred_month)
        order_m = order_m.groupby(['category', 'order_date'])[['ord_qty']].sum()
        order_m.rename(columns={'ord_qty': 'act_ord_qty'})

    @property
    def order(self):
        return self._order

    @property
    def level1_order(self):
        return self._level1_order

    @property
    def category_info(self):
        return self._category_info

    @property
    def cate_aver_price(self):
        return self._cate_aver_price


def main():
    ord_path = "../data/level2/m111-sku-order-all-final.csv"
    dis_path = None
    inv_path = None
    categories = ['CRZJ', 'CRXDG', 'CRYJ', 'JSJ', 'YSJ', 'RR', 'DR', 'CRXWJ']
    odi_data = Level1OdiData(ord_path, dis_path, inv_path, categories)
    X_train, y_train, X_test = odi_data.prepare_data_for_predict(4, left_bound='2016-04')
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)


if __name__ == '__main__':
    main()
