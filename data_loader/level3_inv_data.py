# _*_ coding: utf-8 _*_

"""
Prepare level-3 inventory data.

Author: Genpeng Xu
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Own Customized modules
from util.feature_util import *
from util.data_util import transform_channel
from util.date_util import get_curr_date, get_days_of_month
from global_vars import INV_DATA_DIR, INV_DATA_COLUMN_NAMES


class Level3InvData:
    """Invertory data of Level-3 (per sku per customer)."""

    def __init__(self, categories=None, need_unitize=True):
        self._curr_year, self._curr_month, _ = get_curr_date()
        self._inv_data_path = self._get_data_path()
        self._inv = self._load_inv_data(categories, need_unitize)
        self._inv_cus_sku_month = self._get_inv_data_per_cus_sku()  # 得到每个代理商每个SKU每个月库存
        self._customer_info, self._customer_info_encoded = self._get_cus_info()  # 得到代理商的信息
        self._sku_info, self._sku_info_encoded = self._get_sku_info()  # 得到SKU的信息
        self._inv_sku_month = self._get_inv_data_per_sku()  # 得到每个SKU每个月的库存
        self._inv_cus_cate1_month = self._get_inv_data_per_cus_cate1()  # 得到每个代理商每个大类的库存
        self._inv_cus_cate2_month = self._get_inv_data_per_cus_cate2()  # 得到每个代理商每个小类的库存
        self._all_cates = categories if categories else list(self._inv.first_cate_name.unique())

    def _get_data_path(self):
        date_flag = "%d%02d" % (self._curr_year, self._curr_month)
        filename = "m111-inv_%s.txt" % date_flag
        return os.path.join(INV_DATA_DIR, date_flag, filename)

    def _load_inv_data(self, categories=None, need_unitize=True):
        inv = pd.read_csv(INV_DATA_DIR, sep='\u001e', header=None,
                          names=INV_DATA_COLUMN_NAMES, parse_dates=[0])
        if categories:
            inv = inv.loc[inv.first_cate_code.isin(categories)]
        inv = inv.sort_values(by='order_date').reset_index(drop=True)
        if need_unitize:  # 是否需要单位化，即以万作为单位
            inv['inv_qty'] = inv.inv_qty / 10000
            inv['inv_amount'] = inv.inv_amount / 10000
        inv['channel_name'] = inv.channel_name.apply(lambda x: transform_channel(x))
        inv['sales_chan_name'] = inv.sales_chan_name.str.replace(r'\\N', '未知')
        return inv

    def _get_inv_data_per_cus_sku(self):
        """Get monthly inventory data per customer per sku."""
        tmp = self._inv.copy()
        tmp['order_month'] = tmp.order_date.astype(str).apply(lambda x: x[:7])
        tmp = tmp.groupby(['customer_code', 'item_code', 'order_month'])[['inv_qty']].sum()
        tmp['inv_qty'] = tmp.inv_qty.apply(lambda x: x if x > 0 else 0)
        inv_cus_sku_month = tmp.unstack(level=-1).fillna(0)
        end_dt_str = '%d-%02d-%d' % (self._curr_year,
                                     self._curr_month,
                                     get_days_of_month(self._curr_year, self._curr_month))
        inv_cus_sku_month.columns = pd.date_range(start='2017-04-30', end=end_dt_str, freq='M')
        return inv_cus_sku_month

    def _get_inv_data_per_sku(self):
        """Get monthly inventory data per sku."""
        inv_sku_month = self._inv_cus_sku_month.groupby(['item_code'])[self._inv_cus_sku_month.columns].sum()
        inv_sku_month = inv_sku_month.reindex(self._inv_cus_sku_month.index.get_level_values(1))
        return inv_sku_month

    def _get_cus_info(self):
        """Get information of all customers."""
        label_enc = LabelEncoder()
        customer_info = self._inv.drop_duplicates(['customer_code'], keep='last')
        customer_info = customer_info[['customer_code', 'customer_name', 'sales_cen_code',
                                       'sales_cen_name', 'sales_region_name', 'province',
                                       'city', 'district', 'customer_type', 'is_usable',
                                       'channel_name', 'channel_level', 'sales_chan_name']]
        customer_info.set_index('customer_code', inplace=True)
        customer_info['sales_cen_id'] = label_enc.fit_transform(customer_info['sales_cen_code'].astype(str))
        customer_info['sales_region_id'] = label_enc.fit_transform(customer_info['sales_region_name'])
        customer_info['province_id'] = label_enc.fit_transform(customer_info['province'])
        customer_info['city_id'] = label_enc.fit_transform(customer_info['city'])
        customer_info['district_id'] = label_enc.fit_transform(customer_info['district'])
        customer_info['customer_type'] = label_enc.fit_transform(customer_info['customer_type'])
        customer_info['is_usable'] = label_enc.fit_transform(customer_info['is_usable'])
        customer_info['channel_id'] = label_enc.fit_transform(customer_info['channel_name'])
        customer_info['channel_level'] = label_enc.fit_transform(customer_info['channel_level'])
        customer_info['sales_chan_id'] = label_enc.fit_transform(customer_info['sales_chan_name'])
        customer_info_encoded = customer_info.drop(
            columns=['customer_name', 'sales_cen_code', 'sales_cen_name',
                     'sales_region_name', 'province', 'city', 'district',
                     'channel_name', 'sales_chan_name']
        )
        customer_info_encoded = customer_info_encoded.reindex(self._inv_cus_sku_month.index.get_level_values(0))
        return customer_info, customer_info_encoded

    def _get_sku_info(self):
        """Get information of all SKUs."""
        label_enc = LabelEncoder()
        sku_info = self._inv.drop_duplicates(['item_code'], keep='last')
        sku_info = sku_info[['item_code', 'item_name', 'first_cate_code', 'first_cate_name',
                             'second_cate_code', 'second_cate_name', 'item_price']]
        sku_info.set_index('item_code', inplace=True)
        sku_info['first_cate_id'] = label_enc.fit_transform(sku_info.first_cate_code)
        sku_info['second_cate_id'] = label_enc.fit_transform(sku_info.second_cate_code)
        sku_info_encoded = sku_info.drop(
            columns=['item_name', 'first_cate_code', 'first_cate_name', 'second_cate_code', 'second_cate_name']
        )
        sku_info_encoded = sku_info_encoded.reindex(self._inv_cus_sku_month.index.get_level_values(1))
        return sku_info, sku_info_encoded

    def _get_inv_data_per_cus_cate1(self):
        """Get monthly inventory data per customer per first level category."""
        inv_cus_cate1_month = self._inv_cus_sku_month.reset_index()
        inv_cus_cate1_month['first_cate_id'] = self._sku_info_encoded.first_cate_id.values
        inv_cus_cate1_month_index = inv_cus_cate1_month[['customer_code', 'first_cate_id']]
        inv_cus_cate1_month = inv_cus_cate1_month.groupby(
            ['customer_code', 'first_cate_id'])[self._inv_cus_sku_month.columns].sum()
        inv_cus_cate1_month = inv_cus_cate1_month.reindex(inv_cus_cate1_month_index)
        return inv_cus_cate1_month

    def _get_inv_data_per_cus_cate2(self):
        """Get monthly inventory data per customer per second level category."""
        inv_cus_cate2_month = self._inv_cus_sku_month.reset_index()
        inv_cus_cate2_month['second_cate_id'] = self._sku_info_encoded.second_cate_id.values
        inv_cus_cate2_month_index = inv_cus_cate2_month[['customer_code', 'second_cate_id']]
        inv_cus_cate2_month = inv_cus_cate2_month.groupby(
            ['customer_code', 'second_cate_id'])[self._inv_cus_sku_month.columns].sum()
        inv_cus_cate2_month = inv_cus_cate2_month.reindex(inv_cus_cate2_month_index)
        return inv_cus_cate2_month

    @property
    def curr_year(self):
        return self._curr_year

    @property
    def get_curr_month(self):
        return self._curr_month

    @property
    def inv_data_path(self):
        return self._inv_data_path

    @property
    def all_cates(self):
        return self._all_cates

    @property
    def customer_info(self):
        return self._customer_info

    @property
    def sku_info(self):
        return self._sku_info


def main():
    level3_inv_data = Level3InvData()
    print(level3_inv_data.inv_data_path)


if __name__ == '__main__':
    main()
