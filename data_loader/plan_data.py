# _*_ coding: utf-8 _*_

"""
Plan (demand) data.

Author: Genpeng Xu
"""

import os
import numpy as np
import pandas as pd
from typing import Union

# Own customized variables
from global_vars import PLAN_DATA_DIR, PLAN_DATA_COLUMN_NAMES
from util.feature_util import get_val, get_pre_vals


class PlanData:
    def __init__(self, year, month, need_unitize=True):
        self._year, self._month = year, month
        self._version_flag = "%d-%02d" % (self._year, self._month)
        self._plan = self._get_plan_data(need_unitize)
        self._plan_sku_month = self._get_month_plan_per_sku()
        self._plan_sku_month_mean = self._plan_sku_month.replace(0, np.nan).mean(axis=1)

    def _get_plan_data_filepath(self):
        filename = "m111-plan_%s.txt" % self._version_flag
        return os.path.join(PLAN_DATA_DIR, self._version_flag, filename)

    def _get_plan_data(self, need_unitize=True):
        print("[INFO] Start loading plan data...")
        plan_data_filepath = self._get_plan_data_filepath()
        plan = pd.read_csv(plan_data_filepath, sep='\u001e', header=None, names=PLAN_DATA_COLUMN_NAMES)
        print("[INFO] Loading finished!")
        print("[INFO] Start preprocessing plan data...")
        plan = self._preprocess_plan_data(plan, need_unitize)
        print("[INFO] Preprocessing finished!")
        return plan

    def _preprocess_plan_data(self, plan: pd.DataFrame, need_unitize: bool = True) -> pd.DataFrame:
        plan = plan.loc[plan.bd_code == 'M111']
        plan = plan.loc[plan.plan_mode.isin(['NORMAL', 'CUSTOMER'])]
        plan['item_code'] = plan.item_code.astype(str).apply(lambda x: x.strip())
        plan['period_code'] = pd.to_datetime(plan.period_code, format='%Y%m')
        plan = plan.sort_values(by=['period_code'], ascending=True)
        if need_unitize:
            plan['month_plan_qty'] = plan.month_plan_qty / 10000
        return plan

    def _get_month_plan_per_sku(self):
        temp = self._plan.copy()
        temp['order_date'] = temp.period_code.astype(str).apply(lambda x: x[:7])
        temp = temp.groupby(['item_code', 'order_date'])[['month_plan_qty']].sum()
        plan_sku_month = temp.unstack(level=-1).fillna(0)
        plan_sku_month.columns = pd.date_range(start='2018-12-31', periods=len(plan_sku_month.columns), freq='M')
        return plan_sku_month

    def get_one_month(self,
                      year: int,
                      month: int,
                      need_index: bool = False) -> Union[np.ndarray, pd.Series]:
        return get_val(self._plan_sku_month, year, month, need_index)

    def get_pre_months(self,
                       year: int,
                       month: int,
                       periods: int = 3,
                       need_index: bool = False) -> pd.DataFrame:
        return get_pre_vals(self._plan_sku_month, year, month, periods, need_index)

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
    def plan_sku_month_mean(self):
        return self._plan_sku_month_mean


if __name__ == '__main__':
    plan_data = PlanData(2019, 12)
    print(plan_data.get_pre_months(2019, 12, 3).head())
