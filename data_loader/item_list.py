# _*_ coding: utf-8 _*_

"""
Item list loader.

Author: Genpeng Xu
"""

import pandas as pd
from datetime import date
from bunch import Bunch
from sqlalchemy import create_engine

# Own customized modules
from util.data_util import remove_whitespace
from global_vars import ITEM_LIST_MYSQL_CONFIG, ITEM_LIST_COLUMN_NAMES


class ItemList:
    def __init__(self, year, month):
        self._year, self._month = year, month
        self._version_flag = "%d-%02d" % (year, month) if date(year, month, 1) > date(2019, 10, 1) else '2019-10'
        self._items = self._load_and_preprocess()
        self._white_items = set()
        self._scheduled_delisting_items = set()
        self._delisting_items = set()
        self._new_items = set()
        self._curr_new_items = set()
        self._six_eighteen_main_items = set()
        self._double_eleven_main_items = set()
        self._infer_all_sets()

    def is_white_items(self, item_code):
        return item_code in self._white_items

    def is_scheduled_delisting_items(self, item_code):
        return item_code in self._scheduled_delisting_items

    def is_delisting_items(self, item_code):
        return item_code in self._delisting_items

    def is_new_items(self, item_code):
        return item_code in self._new_items

    def is_curr_new_items(self, item_code):
        return item_code in self._curr_new_items

    def is_six_eighteen_main_items(self, item_code):
        return item_code in self._six_eighteen_main_items

    def is_double_eleven_main_items(self, item_code):
        return item_code in self._double_eleven_main_items

    def _load_and_preprocess(self):
        # read in item list data
        config = Bunch(ITEM_LIST_MYSQL_CONFIG)
        config_str = "mysql+pymysql://%s:%s@%s:%s/%s" % (config.mode,
                                                         config.user,
                                                         config.host,
                                                         config.port,
                                                         config.database)
        engine = create_engine(config_str)
        sql = "SELECT * FROM m111_item_list WHERE version = '%s'" % self._version_flag
        items = pd.read_sql_query(sql, engine)
        items.columns = ITEM_LIST_COLUMN_NAMES
        # preprocess item list data
        str_cols = [
            'item_code', 'item_name', 'first_cate_code', 'state',
            'manu_code', 'need_calc', 'is_highend', 'appear_date',
            'delisting_date', 'scheduled_delisting_date',
            'six_eighteen_main_product_date', 'double_eleven_main_product_date'
        ]
        remove_whitespace(items, str_cols)
        dt_cols = ['appear_date', 'delisting_date', 'scheduled_delisting_date',
                   'six_eighteen_main_product_date', 'double_eleven_main_product_date']
        for col in dt_cols:
            items[col] = items[col].apply(lambda x: x[:7])
            items[col] = pd.to_datetime(items[col], format='%Y-%m')
        return items

    def _infer_all_sets(self):
        self._white_items = self._get_white_items()
        self._scheduled_delisting_items = self._get_scheduled_delisting_items()
        self._delisting_items = self._get_delisting_items()
        self._new_items = self._get_new_items()
        self._curr_new_items = self._get_curr_new_items()
        self._six_eighteen_main_items = self._get_six_eighteen_main_items()
        self._double_eleven_main_items = self._get_double_eleven_main_items()

    def _get_white_items(self):
        white_items = set(self._items.loc[self._items.need_calc == '是'].item_code)
        return white_items

    def _get_scheduled_delisting_items(self,):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.scheduled_delisting_date.isna()]
        tmp['interval'] = tmp.scheduled_delisting_date.apply(
            lambda x: pd.date_range(end=x.date(), periods=2, freq=pd.offsets.MonthBegin(1))
        )
        tmp['is_in'] = tmp.interval.apply(lambda x: '%d-%02d-01' % (self._year, self._month) in x)
        tmp = tmp.loc[tmp.is_in]
        scheduled_delisting_items = set(tmp.item_code)
        return scheduled_delisting_items

    def _get_delisting_items(self):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.delisting_date.isna()]
        tmp['interval'] = tmp.delisting_date.apply(
            lambda x: pd.date_range(end=x.date(), periods=4, freq=pd.offsets.MonthBegin(1))
        )
        tmp['is_in'] = tmp.interval.apply(lambda x: '%d-%02d-01' % (self._year, self._month) in x)
        tmp = tmp.loc[tmp.is_in]
        delisting_items = set(tmp.item_code)
        return delisting_items

    def _get_new_items(self):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.appear_date.isna()]
        tmp['interval'] = tmp.appear_date.apply(
            lambda x: pd.date_range(start=x.date(), periods=3, freq=pd.offsets.MonthBegin(1))
        )
        tmp['is_in'] = tmp.interval.apply(lambda x: '%d-%02d-01' % (self._year, self._month) in x)
        tmp = tmp.loc[tmp.is_in]
        new_items = set(tmp.item_code)
        return new_items

    def _get_curr_new_items(self):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.appear_date.isna()]
        tmp = tmp.loc[tmp.appear_date == '%d-%02d-01' % (self._year, self._month)]
        curr_new_items = set(tmp.item_code)
        return curr_new_items

    def _get_six_eighteen_main_items(self):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.six_eighteen_main_product_date.isna()]
        tmp['interval'] = tmp.six_eighteen_main_product_date.apply(
            lambda x: pd.date_range(start=x.date(), periods=4, freq=pd.offsets.MonthBegin(1))
        )
        tmp['is_in'] = tmp.interval.apply(lambda x: '%d-%02d-01' % (self._year, self._month) in x)
        tmp = tmp.loc[tmp.is_in]
        six_eighteen_main_items = set(tmp.item_code)
        return six_eighteen_main_items

    def _get_double_eleven_main_items(self):
        tmp = self._items.copy()
        tmp = tmp.loc[~tmp.double_eleven_main_product_date.isna()]
        tmp['interval'] = tmp.double_eleven_main_product_date.apply(
            lambda x: pd.date_range(start=x.date(), periods=4, freq=pd.offsets.MonthBegin(1))
        )
        tmp['is_in'] = tmp.interval.apply(lambda x: '%d-%02d-01' % (self._year, self._month) in x)
        tmp = tmp.loc[tmp.is_in]
        double_eleven_main_items = set(tmp.item_code)
        return double_eleven_main_items

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def version_flag(self):
        return self._version_flag

    @property
    def items(self):
        return self._items

    @property
    def white_items(self):
        return self._white_items

    @property
    def scheduled_delisting_items(self):
        return self._scheduled_delisting_items

    @property
    def delisting_items(self):
        return self._delisting_items

    @property
    def new_items(self):
        return self._new_items

    @property
    def curr_new_items(self):
        return self._curr_new_items

    @property
    def six_eighteen_main_items(self):
        return self._six_eighteen_main_items

    @property
    def double_eleven_main_items(self):
        return self._double_eleven_main_items


def _test():
    item_list = ItemList(2019, 10)
    print(len(item_list.scheduled_delisting_items))
    print(len(item_list.delisting_items))
    print(len(item_list.new_items))
    print(len(item_list.curr_new_items))
    print(len(item_list.six_eighteen_main_items))
    print(len(item_list.double_eleven_main_items))


if __name__ == '__main__':
    _test()
