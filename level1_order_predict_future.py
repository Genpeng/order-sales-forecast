# _*_ coding: utf-8 _*_

"""
Predict order future result of Level-1.

Author: Genpeng Xu
"""

import gc
import time
import warnings
import numpy as np
import pandas as pd
from bunch import Bunch
from datetime import datetime
from typing import Union, List

# Own customized modules
from global_vars import (SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG,
                         UAT_ESB_URL, PROD_ESB_URL, CATE_CODE_2_CATE_NAME)
from data_loader.level1_data import Level1DataLoader
from infer.sales_infer import RFSalesInfer
from writer.kudu_result_writer import KuduResultWriter
from util.feature_util import modify_training_set
from util.config_util import get_args, process_config
from util.date_util import (get_curr_date, infer_month,
                            get_pre_months, timestamp_to_time)

warnings.filterwarnings('ignore')


def update_future_for_level1_order(model_config: Bunch,
                                   db_config: Bunch,
                                   start_pred_year: int,
                                   start_pred_month: int,
                                   periods: int = 4,
                                   categories: Union[str, List[str]] = 'all',
                                   need_unitize: bool = True) -> None:
    """Update order future result of level1."""

    # Step 1: Read in data
    # ============================================================================================ #

    level1_data = Level1DataLoader(start_pred_year, start_pred_month,
                                   categories=categories, need_unitize=need_unitize, label_data='order')

    # Step 2: Training and predicting
    # ============================================================================================ #

    year_upper_bound, month_upper_bound = infer_month(start_pred_year, start_pred_month, offset=-periods)
    train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2016-03')

    preds_test = []
    for i in range(periods):
        X_train, y_train = level1_data.prepare_training_set(train_months, gap=i)
        X_train, y_train = modify_training_set(X_train, y_train)
        X_test = level1_data.prepare_testing_set(start_pred_year, start_pred_month, gap=i)
        predictor = RFSalesInfer(model_config)
        predictor.fit(X_train, y_train)
        preds_test.append(predictor.predict(X_test))

    # Step 3: Process forecast result & write into "水晶球"
    # ============================================================================================ #

    df_test = level1_data.get_true_order_data(start_pred_year, start_pred_month)
    df_pred_test = level1_data.add_index(preds_test, start_pred_year, start_pred_month)
    df_pred_test_more = level1_data.predict_by_history(start_pred_year, start_pred_month, gap=periods)
    df_pred_test = pd.concat(
        [df_pred_test, df_pred_test_more], axis=1
    ).stack().to_frame('pred_ord_qty')
    df_pred_test.index.set_names(['first_cate_code', 'order_date'], inplace=True)
    if need_unitize:
        df_pred_test['pred_ord_qty'] = df_pred_test.pred_ord_qty.apply(lambda x: x if x > 0 else 0.0025)
    else:
        df_pred_test['pred_ord_qty'] = df_pred_test.pred_ord_qty.apply(lambda x: x if x > 0 else 25)
    df_pred_test['pred_ord_qty'] = np.round(df_pred_test.pred_ord_qty, decimals=4 if need_unitize else 0)

    result = df_pred_test.join(df_test, how='left').reset_index()
    result.act_ord_qty.fillna(0, inplace=True)

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    result['first_cate_name'] = result.first_cate_code.map(CATE_CODE_2_CATE_NAME)
    cate_info_dict = level1_data.cate_info.to_dict()
    result['aver_price'] = result.first_cate_code.map(cate_info_dict['cate_aver_price'])

    result['act_ord_amount'] = np.round(result.act_ord_qty * result.aver_price, decimals=4 if need_unitize else 0)
    result['pred_ord_amount'] = np.round(result.pred_ord_qty * result.aver_price, decimals=4 if need_unitize else 0)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    if db_config.env == 'SIT':
        kudu_config = SIT_DB_CONFIG
    elif db_config.env == 'UAT':
        kudu_config = UAT_DB_CONFIG
    elif db_config.env == 'PROD':
        kudu_config = PROD_DB_CONFIG
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")

    writer = KuduResultWriter(Bunch(kudu_config))
    writer.clear_months_after(db_config.table1_name, 'order_date', start_pred_year, start_pred_month)
    writer.upsert(result, db_config.table1_name, db_config.batch_size)

    del result
    gc.collect()

    # Step 4: Process forecast result & write into "明细表"
    # ============================================================================================ #

    result = level1_data.add_index_v2(preds_test[1:])
    if need_unitize:
        for col in result.columns:
            result[col] = result[col].apply(lambda x: 0.0025 if x < 0 else x)
    else:
        for col in result.columns:
            result[col] = result[col].apply(lambda x: 25 if x < 0 else x)
    result = result.reset_index()

    result['bu_code'] = '30015305'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    m1_year, m1_month = infer_month(start_pred_year, start_pred_month, 1)
    result['order_date'] = "%d%02d" % (m1_year, m1_month)

    result['first_cate_name'] = result.first_cate_code.map(CATE_CODE_2_CATE_NAME)
    result['aver_price'] = result.first_cate_code.map(cate_info_dict['cate_aver_price'])

    result['pred_ord_amount_m1'] = np.round(result.pred_ord_qty_m1 * result.aver_price,
                                            decimals=4 if need_unitize else 0)
    result['pred_ord_amount_m2'] = np.round(result.pred_ord_qty_m2 * result.aver_price,
                                            decimals=4 if need_unitize else 0)
    result['pred_ord_amount_m3'] = np.round(result.pred_ord_qty_m3 * result.aver_price,
                                            decimals=4 if need_unitize else 0)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    if need_unitize:
        result['pred_ord_qty_m1'] = np.round(result.pred_ord_qty_m1 * 10000)
        result['pred_ord_qty_m2'] = np.round(result.pred_ord_qty_m2 * 10000)
        result['pred_ord_qty_m3'] = np.round(result.pred_ord_qty_m3 * 10000)
        result['pred_ord_amount_m1'] = np.round(result.pred_ord_amount_m1 * 10000)
        result['pred_ord_amount_m2'] = np.round(result.pred_ord_amount_m2 * 10000)
        result['pred_ord_amount_m3'] = np.round(result.pred_ord_amount_m3 * 10000)

    writer = KuduResultWriter(Bunch(kudu_config))
    writer.clear_one_month(db_config.table2_name, 'order_date', m1_year, m1_month)
    writer.upsert(result, db_config.table2_name, db_config.batch_size)


if __name__ == '__main__':
    # Load & parse configuration file
    # ============================================================================================ #

    print("[INFO] Start loading & parsing configuration...")
    parser, config = None, None
    try:
        args, parser = get_args()  # get the path of configuration file
        config = process_config(args.config)
    except Exception as e:
        print(e)
        if parser:
            parser.print_help()
        exit(0)
    print("[INFO] Parsing finished!")

    # Update future result of level1 order
    # ============================================================================================ #

    curr_year, curr_month, _ = get_curr_date()
    if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
        raise Exception("[INFO] The data is not ready yet, please try again after 13:00 on the 16th!")
    if config.periods < 2:
        raise Exception("[INFO] The predicted period is less than 2!!!")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.db_config)
    update_future_for_level1_order(model_config=model_config,
                                   db_config=db_config,
                                   start_pred_year=curr_year,
                                   start_pred_month=curr_month,
                                   periods=config.periods,
                                   categories=config.categories,
                                   need_unitize=config.need_unitize)
