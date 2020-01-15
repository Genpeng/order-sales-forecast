# _*_ coding: utf-8 _*_

"""
Predict distribution future result of Level-3.

Author: Genpeng Xu
"""

import time
import numpy as np
import pandas as pd
from bunch import Bunch
from datetime import datetime
from typing import Union, List

# Own customized variables & modules
from global_vars import SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG
from data_loader.level3_dis_data import Level3DisDataLoader
from data_loader.item_list import ItemList
from data_loader.customer_list import CustomerList
from infer.sales_infer import LGBMSalesInfer
from writer.kudu_result_writer import KuduResultWriter
from util.feature_util import modify_training_set
from util.config_util import get_args, process_config
from util.date_util import get_curr_date, infer_month, get_pre_months, timestamp_to_time


def update_future_for_level3_dis(model_config: Bunch,
                                 db_config: Bunch,
                                 start_pred_year: int,
                                 start_pred_month: int,
                                 periods: int = 4,
                                 categories: Union[str, List[str]] = 'all',
                                 need_unitize: bool = True) -> None:
    """Update distribution future result of level3."""

    # Step 1: Read in data
    # ============================================================================================ #

    data_loader = Level3DisDataLoader(start_pred_year, start_pred_month,
                                      categories=categories, need_unitize=need_unitize)

    # Step 2: Training and predicting
    # ============================================================================================ #

    year_upper_bound, month_upper_bound = infer_month(start_pred_year, start_pred_month, offset=-periods)
    train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2018-07')

    preds_test = []
    for i in range(periods):
        X_train, y_train = data_loader.prepare_training_set(train_months, gap=i)
        X_train, y_train = modify_training_set(X_train, y_train)
        X_test = data_loader.prepare_testing_set(start_pred_year, start_pred_month, gap=i)
        predictor = LGBMSalesInfer(model_config)
        predictor.fit(X_train, y_train)
        preds_test.append(predictor.predict(X_test))

    # Step 3: Process forecast result
    # ============================================================================================ #

    df_test = data_loader.get_true_data(start_pred_year, start_pred_month)
    df_preds_test = data_loader.add_index(preds_test, start_pred_year, start_pred_month)
    df_preds_test_more = data_loader.predict_by_history(start_pred_year, start_pred_month, gap=periods)
    df_preds_test = pd.concat(
        [df_preds_test, df_preds_test_more], axis=1
    ).stack().to_frame('pred_dis_qty')
    df_preds_test.index.set_names(['customer_code', 'item_code', 'order_date'], inplace=True)
    df_preds_test['pred_dis_qty'] = df_preds_test.pred_dis_qty.apply(lambda x: x if x > 0 else 0)
    df_preds_test['pred_dis_qty'] = np.round(df_preds_test.pred_dis_qty, decimals=4 if need_unitize else 0)

    result = df_preds_test.join(df_test, how='left').reset_index()
    result.act_dis_qty.fillna(0, inplace=True)

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    customer_info_dict = data_loader.customer_info.to_dict()
    result['customer_name'] = result.customer_code.map(customer_info_dict['customer_name'])
    result['sales_cen_code'] = result.customer_code.map(customer_info_dict['sales_cen_code'])
    result['sales_cen_name'] = result.customer_code.map(customer_info_dict['sales_cen_name'])
    # result['province_code'] = result.customer_code.map(customer_info['province_id'])
    # result['city_code'] = result.customer_code.map(customer_info['city_id'])
    # result['district_code'] = result.customer_code.map(customer_info['district_id'])
    # result['channel_code'] = result.customer_code.map(customer_info['channel_name_id'])
    result['province_name'] = result.customer_code.map(customer_info_dict['province'])
    result['city_name'] = result.customer_code.map(customer_info_dict['city'])
    result['district_name'] = result.customer_code.map(customer_info_dict['district'])

    sku_info_dict = data_loader.sku_info.to_dict()
    result['item_name'] = result.item_code.map(sku_info_dict['item_name'])
    result['first_cate_code'] = result.item_code.map(sku_info_dict['first_cate_code'])
    result['second_cate_code'] = result.item_code.map(sku_info_dict['second_cate_code'])
    result['first_cate_name'] = result.item_code.map(sku_info_dict['first_cate_name'])
    result['second_cate_name'] = result.item_code.map(sku_info_dict['second_cate_name'])
    result['channel_name'] = result.item_code.map(sku_info_dict['channel_name'])
    result['item_price'] = result.item_code.map(sku_info_dict['item_price'])

    result['act_dis_amount'] = np.round(result.act_dis_qty * result.item_price, decimals=4 if need_unitize else 0)
    result['pred_dis_amount'] = np.round(result.pred_dis_qty * result.item_price, decimals=4 if need_unitize else 0)
    result['dis_pred_time'] = timestamp_to_time(time.time())

    customer_list = CustomerList()
    item_list = ItemList(start_pred_year, start_pred_month)
    result = result.loc[result.customer_code.apply(lambda x: customer_list.is_white_customer(x))]
    result = result.loc[result.item_code.apply(lambda x: item_list.is_white_items(x))]

    # Step 4: Write into database (Kudu)
    # ============================================================================================ #

    if db_config.env == 'SIT':
        writer = KuduResultWriter(Bunch(SIT_DB_CONFIG))
    elif db_config.env == 'UAT':
        writer = KuduResultWriter(Bunch(UAT_DB_CONFIG))
    elif db_config.env == 'PROD':
        writer = KuduResultWriter(Bunch(PROD_DB_CONFIG))
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")

    writer.clear_months_after(db_config.table_name, 'order_date', start_pred_year, start_pred_month)
    writer.upsert(result, db_config.table_name, db_config.batch_size)


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

    # Update future result of level3 distribution
    # ============================================================================================ #

    curr_year, curr_month, _ = get_curr_date()
    if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
        raise Exception("[INFO] The data is not ready yet, please try again after 13:00 on the 16th!")
    if config.periods < 2:
        raise Exception("[INFO] The predicted period is less than 2!!!")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.db_config)
    update_future_for_level3_dis(model_config=model_config,
                                 db_config=db_config,
                                 start_pred_year=curr_year,
                                 start_pred_month=curr_month,
                                 periods=config.periods,
                                 categories=config.categories,
                                 need_unitize=config.need_unitize)
