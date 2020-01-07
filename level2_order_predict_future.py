# _*_ coding: utf-8 _*_

"""
Predict order future result of Level-2.

Author: Genpeng Xu
"""

import gc
import time
import numpy as np
import pandas as pd
from bunch import Bunch
from datetime import datetime
from typing import Union, List

# Own customized modules
from global_vars import (SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG,
                         UAT_ESB_URL, PROD_ESB_URL)
from data_loader.level2_data import Level2DataLoader
from data_loader.item_list import ItemList
from data_loader.plan_data import PlanData
from infer.sales_infer import LGBMSalesInfer
from writer.kudu_result_writer import KuduResultWriter
from util.feature_util import modify_training_set, rule_func
from util.config_util import get_args, process_config
from util.date_util import (get_curr_date, infer_month, get_pre_months,
                            timestamp_to_time, get_days_of_month)
from util.esb_util import push_to_esb


def update_future_for_level2_order(model_config: Bunch,
                                   db_config: Bunch,
                                   start_pred_year: int,
                                   start_pred_month: int,
                                   periods: int = 4,
                                   categories: Union[str, List[str]] = 'all',
                                   need_unitize: bool = True) -> None:
    """Update order future result of level2."""

    # Step 1: Read in data
    # ============================================================================================ #

    level2_data = Level2DataLoader(start_pred_year, start_pred_month,
                                   categories=categories, need_unitize=need_unitize, label_data='order')
    plan_data = PlanData(start_pred_year, start_pred_month, need_unitize=need_unitize)
    item_list = ItemList(start_pred_year, start_pred_month)

    # Step 2: Training and predicting
    # ============================================================================================ #

    year_upper_bound, month_upper_bound = infer_month(start_pred_year, start_pred_month, offset=-periods)
    train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2016-03')

    preds_test = []
    for i in range(periods):
        X_train, y_train = level2_data.prepare_training_set(train_months, gap=i)
        X_train, y_train = modify_training_set(X_train, y_train)
        X_test = level2_data.prepare_testing_set(start_pred_year, start_pred_month, gap=i)
        predictor = LGBMSalesInfer(model_config)
        predictor.fit(X_train, y_train)
        preds_test.append(predictor.predict(X_test))

    # Step 3: Process forecast result & write into "水晶球"
    # ============================================================================================ #

    df_test = level2_data.get_true_order_data(start_pred_year, start_pred_month)
    df_pred_test = level2_data.add_index(preds_test, start_pred_year, start_pred_month)
    df_pred_test_more = level2_data.predict_by_history(start_pred_year, start_pred_month, gap=periods)
    df_pred_test = pd.concat(
        [df_pred_test, df_pred_test_more], axis=1
    ).stack().to_frame('pred_ord_qty')
    df_pred_test.index.set_names(['item_code', 'order_date'], inplace=True)
    if need_unitize:
        df_pred_test['pred_ord_qty'] = df_pred_test.pred_ord_qty.apply(lambda x: x if x > 0 else 0.0025)
    else:
        df_pred_test['pred_ord_qty'] = df_pred_test.pred_ord_qty.apply(lambda x: x if x > 0 else 25)
    df_pred_test['pred_ord_qty'] = np.round(df_pred_test.pred_ord_qty, decimals=4 if need_unitize else 0)

    result = df_pred_test.join(df_test, how='left').reset_index()
    result.act_ord_qty.fillna(0, inplace=True)

    ## 修正区 ##

    sku_info_dict = level2_data.sku_info.to_dict()

    m1_year, m1_month = infer_month(start_pred_year, start_pred_month, 1)
    m1_res = result.loc[result.order_date == "%d%02d" % (m1_year, m1_month)]
    other_res = result.loc[~(result.order_date == "%d%02d" % (m1_year, m1_month))]

    rule_res = m1_res.copy()
    order_sku_month_pre6_mean = level2_data.get_pre_order_vals(
        start_pred_year, start_pred_month, 6, True).replace(0, np.nan).mean(axis=1)
    order_sku_month_pre1 = level2_data.get_pre_order_vals(
        start_pred_year, start_pred_month, 1, True).mean(axis=1)
    dis_sku_month_pre3_mean = level2_data.get_pre_dis_vals(
        start_pred_year, start_pred_month, 3, True).replace(0, np.nan).mean(axis=1)
    dis_sku_month_pre1 = level2_data.get_pre_dis_vals(
        start_pred_year, start_pred_month, 1, True).mean(axis=1)
    plan_sku_month_mean = plan_data.plan_sku_month_mean

    rule_res['ord_sku_month_pre6_mean'] = rule_res.item_code.map(order_sku_month_pre6_mean)
    rule_res['ord_sku_month_pre1'] = rule_res.item_code.map(order_sku_month_pre1)
    rule_res['dis_sku_month_pre3_mean'] = rule_res.item_code.map(dis_sku_month_pre3_mean)
    rule_res['dis_sku_month_pre1'] = rule_res.item_code.map(dis_sku_month_pre1)
    rule_res['plan_sku_month_mean'] = rule_res.item_code.map(plan_sku_month_mean)

    rule_res['is_aver_ord_na'] = (rule_res.ord_sku_month_pre6_mean.isna()) * 1
    rule_res['is_aver_dis_na'] = (rule_res.dis_sku_month_pre3_mean.isna()) * 1
    rule_res['is_aver_plan_na'] = (rule_res.plan_sku_month_mean.isna()) * 1
    rule_res['is_ord_pre1_na'] = (rule_res.ord_sku_month_pre1.isna()) * 1
    rule_res['is_dis_pre1_na'] = (rule_res.dis_sku_month_pre1.isna()) * 1

    rule_res['online_offline_flag'] = rule_res.item_code.map(sku_info_dict['sales_chan_name']).fillna('未知')
    rule_res['project_flag'] = rule_res.item_code.map(sku_info_dict['project_flag']).fillna('未知')

    order_sku_month_pre24_mean = level2_data.get_pre_order_vals(
        start_pred_year, start_pred_month, 24, True).replace(0, np.nan).mean(axis=1)
    curr_new_items = set(order_sku_month_pre24_mean.loc[order_sku_month_pre24_mean.isna()].index)

    dis_sku_month_pre3 = level2_data.get_pre_dis_vals(start_pred_year, start_pred_month, 3, True)
    dis_sku_month_pre3['num_not_null'] = ((dis_sku_month_pre3 > 0) * 1).sum(axis=1)
    new_items_by_dis = set(
        dis_sku_month_pre3.loc[(dis_sku_month_pre3.num_not_null == 1) & (dis_sku_month_pre3.iloc[:, 2] > 0)].index)

    demand = plan_data.get_one_month(m1_year, m1_month, True)
    rule_res['demand'] = rule_res.item_code.map(demand)
    rule_res['is_curr_new'] = rule_res.item_code.apply(lambda x: 1 if x in curr_new_items else 0)
    rule_res['is_new_by_dis'] = rule_res.item_code.apply(lambda x: 1 if x in new_items_by_dis else 0)
    rule_res['demand_dis_ratio'] = rule_res.demand / rule_res.dis_sku_month_pre3_mean

    rule_res['pred_ord_qty_rule'] = rule_res.apply(rule_func, axis=1)
    rule_res['pred_ord_qty_rule'] = rule_res.pred_ord_qty_rule.replace(np.nan, 0)
    rule_res['pred_ord_qty_rule'] = rule_res.apply(
        lambda x: x.pred_ord_qty if x.pred_ord_qty_rule == 0 else x.pred_ord_qty_rule,
        axis=1
    )

    m1_res['pred_ord_qty'] = m1_res.pred_ord_qty * 0.5 + rule_res.pred_ord_qty_rule * 0.5
    result = pd.concat([m1_res, other_res], axis=0)

    ###########

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    sku_info_dict = level2_data.sku_info.to_dict()
    result['item_name'] = result.item_code.map(sku_info_dict['item_name'])
    result['first_cate_code'] = result.item_code.map(sku_info_dict['first_cate_code'])
    result['second_cate_code'] = result.item_code.map(sku_info_dict['second_cate_code'])
    result['first_cate_name'] = result.item_code.map(sku_info_dict['first_cate_name'])
    result['second_cate_name'] = result.item_code.map(sku_info_dict['second_cate_name'])
    result['channel_name'] = result.item_code.map(sku_info_dict['channel_name'])
    result['item_price'] = result.item_code.map(sku_info_dict['item_price'])

    result['act_ord_amount'] = np.round(result.act_ord_qty * result.item_price, decimals=4)
    result['pred_ord_amount'] = np.round(result.pred_ord_qty * result.item_price, decimals=4)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    result = result.loc[result.item_code.apply(lambda x: item_list.is_white_items(x))]

    if db_config.env == 'SIT':
        kudu_config = SIT_DB_CONFIG
        esb_url = UAT_ESB_URL
    elif db_config.env == 'UAT':
        kudu_config = UAT_DB_CONFIG
        esb_url = UAT_ESB_URL
    elif db_config.env == 'PROD':
        kudu_config = PROD_DB_CONFIG
        esb_url = PROD_ESB_URL
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")

    writer = KuduResultWriter(Bunch(kudu_config))
    writer.clear_months_after(db_config.table1_name, 'order_date', start_pred_year, start_pred_month)
    writer.upsert(result, db_config.table1_name, db_config.batch_size)

    del result
    gc.collect()

    # Step 4: Process forecast result & write into "明细表"
    # ============================================================================================ #

    result = level2_data.add_index_v2(preds_test[1:])
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
    result['sales_type'] = "内销"
    result['forecast_type'] = "内销整机预测"

    # m1_year, m1_month = infer_month(start_pred_year, start_pred_month, 1)
    result['order_date'] = "%d%02d" % (m1_year, m1_month)

    sku_info_dict = level2_data.sku_info.to_dict()
    result['item_name'] = result.item_code.map(sku_info_dict['item_name'])
    result['first_cate_code'] = result.item_code.map(sku_info_dict['first_cate_code'])
    result['second_cate_code'] = result.item_code.map(sku_info_dict['second_cate_code'])
    result['first_cate_name'] = result.item_code.map(sku_info_dict['first_cate_name'])
    result['second_cate_name'] = result.item_code.map(sku_info_dict['second_cate_name'])
    result['item_price'] = result.item_code.map(sku_info_dict['item_price'])

    item_list_dict = item_list.items.copy().set_index('item_code').to_dict()
    print(len(item_list_dict['manu_code']))
    result['manu_code'] = result.item_code.map(item_list_dict['manu_code']).fillna('')
    result['area_name'] = ''

    # rule_res = result.copy()
    # rule_res['pred_ord_qty'] = rule_res['pred_ord_qty_m1']
    # order_sku_month_pre6_mean = level2_data.get_pre_order_vals(
    #     start_pred_year, start_pred_month, 6, True).replace(0, np.nan).mean(axis=1)
    # order_sku_month_pre1 = level2_data.get_pre_order_vals(
    #     start_pred_year, start_pred_month, 1, True).mean(axis=1)
    # dis_sku_month_pre3_mean = level2_data.get_pre_dis_vals(
    #     start_pred_year, start_pred_month, 3, True).replace(0, np.nan).mean(axis=1)
    # dis_sku_month_pre1 = level2_data.get_pre_dis_vals(
    #     start_pred_year, start_pred_month, 1, True).mean(axis=1)
    # plan_sku_month_mean = plan_data.plan_sku_month_mean
    #
    # rule_res['ord_sku_month_pre6_mean'] = rule_res.item_code.map(order_sku_month_pre6_mean)
    # rule_res['ord_sku_month_pre1'] = rule_res.item_code.map(order_sku_month_pre1)
    # rule_res['dis_sku_month_pre3_mean'] = rule_res.item_code.map(dis_sku_month_pre3_mean)
    # rule_res['dis_sku_month_pre1'] = rule_res.item_code.map(dis_sku_month_pre1)
    # rule_res['plan_sku_month_mean'] = rule_res.item_code.map(plan_sku_month_mean)
    #
    # rule_res['is_aver_ord_na'] = (rule_res.ord_sku_month_pre6_mean.isna()) * 1
    # rule_res['is_aver_dis_na'] = (rule_res.dis_sku_month_pre3_mean.isna()) * 1
    # rule_res['is_aver_plan_na'] = (rule_res.plan_sku_month_mean.isna()) * 1
    # rule_res['is_ord_pre1_na'] = (rule_res.ord_sku_month_pre1.isna()) * 1
    # rule_res['is_dis_pre1_na'] = (rule_res.dis_sku_month_pre1.isna()) * 1
    #
    # rule_res['online_offline_flag'] = rule_res.item_code.map(sku_info_dict['sales_chan_name']).fillna('未知')
    # rule_res['project_flag'] = rule_res.item_code.map(sku_info_dict['project_flag']).fillna('未知')
    #
    # order_sku_month_pre24_mean = level2_data.get_pre_order_vals(
    #     start_pred_year, start_pred_month, 24, True).replace(0, np.nan).mean(axis=1)
    # curr_new_items = set(order_sku_month_pre24_mean.loc[order_sku_month_pre24_mean.isna()].index)
    #
    # dis_sku_month_pre3 = level2_data.get_pre_dis_vals(start_pred_year, start_pred_month, 3, True)
    # dis_sku_month_pre3['num_not_null'] = ((dis_sku_month_pre3 > 0) * 1).sum(axis=1)
    # new_items_by_dis = set(
    #     dis_sku_month_pre3.loc[(dis_sku_month_pre3.num_not_null == 1) & (dis_sku_month_pre3.iloc[:, 2] > 0)].index)
    #
    # demand = plan_data.get_one_month(m1_year, m1_month, True)
    # rule_res['demand'] = rule_res.item_code.map(demand)
    # rule_res['is_curr_new'] = rule_res.item_code.apply(lambda x: 1 if x in curr_new_items else 0)
    # rule_res['is_new_by_dis'] = rule_res.item_code.apply(lambda x: 1 if x in new_items_by_dis else 0)
    # rule_res['demand_dis_ratio'] = rule_res.demand / rule_res.dis_sku_month_pre3_mean
    #
    # rule_res['pred_ord_qty_rule'] = rule_res.apply(rule_func, axis=1)
    # rule_res['pred_ord_qty_rule'] = rule_res.pred_ord_qty_rule.replace(np.nan, 0)
    # rule_res['pred_ord_qty_rule'] = rule_res.apply(
    #     lambda x: x.pred_ord_qty if x.pred_ord_qty_rule == 0 else x.pred_ord_qty_rule,
    #     axis=1
    # )

    result['pred_ord_qty_m1'] = result.pred_ord_qty_m1 * 0.5 + rule_res.pred_ord_qty_rule * 0.5
    result['avg_dis'] = rule_res['dis_sku_month_pre3_mean'].fillna(0.0)
    result['pred_ord_amount_m1'] = np.round(result.pred_ord_qty_m1 * result.item_price,
                                            decimals=4 if need_unitize else 0)
    result['pred_ord_amount_m2'] = np.round(result.pred_ord_qty_m2 * result.item_price,
                                            decimals=4 if need_unitize else 0)
    result['pred_ord_amount_m3'] = np.round(result.pred_ord_qty_m3 * result.item_price,
                                            decimals=4 if need_unitize else 0)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    if need_unitize:
        result['avg_dis'] = np.round(result.avg_dis * 10000)
        result['pred_ord_qty_m1'] = np.round(result.pred_ord_qty_m1 * 10000)
        result['pred_ord_qty_m2'] = np.round(result.pred_ord_qty_m2 * 10000)
        result['pred_ord_qty_m3'] = np.round(result.pred_ord_qty_m3 * 10000)
        result['pred_ord_amount_m1'] = np.round(result.pred_ord_amount_m1 * 10000)
        result['pred_ord_amount_m2'] = np.round(result.pred_ord_amount_m2 * 10000)
        result['pred_ord_amount_m3'] = np.round(result.pred_ord_amount_m3 * 10000)

    result = result.loc[~result.item_code.apply(lambda x: item_list.is_delisting_items(x))]
    result = result.loc[~(result.manu_code == '')]

    writer = KuduResultWriter(Bunch(kudu_config))
    writer.clear_one_month(db_config.table2_name, 'order_date', m1_year, m1_month)
    writer.upsert(result, db_config.table2_name, db_config.batch_size)

    # Step 5: Push to ESB
    # ============================================================================================ #

    result['customer_code'] = ''
    result['attribute1'] = ''
    result['attribute2'] = ''
    result['attribute3'] = ''
    result['attribute4'] = ''
    result['attribute5'] = ''
    result.rename(columns={'manu_code': 'manu_name'}, inplace=True)
    result['order_date'] = "%d-%02d-%02d" % (m1_year,
                                             m1_month,
                                             get_days_of_month(m1_year, m1_month))
    result = result[['bu_code', 'sales_type', 'manu_name',
                     'area_name', 'customer_code', 'order_date',
                     'first_cate_name', 'second_cate_name', 'item_code',
                     'forecast_type', 'avg_dis', 'item_price',
                     'pred_ord_qty_m1', 'pred_ord_qty_m2', 'pred_ord_qty_m3',
                     'attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5']]
    push_to_esb(result, esb_url)


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

    # Update future result of level2 order
    # ============================================================================================ #

    # curr_year, curr_month, _ = get_curr_date()
    curr_year, curr_month, _ = 2019, 12, 10
    # if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
    #     raise Exception("[INFO] The data is not ready yet, please try again after 13:00 on the 16th!")
    # if config.periods < 2:
    #     raise Exception("[INFO] The predicted period is less than 2!!!")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.db_config)
    update_future_for_level2_order(model_config=model_config,
                                   db_config=db_config,
                                   start_pred_year=curr_year,
                                   start_pred_month=curr_month,
                                   periods=config.periods,
                                   categories=config.categories,
                                   need_unitize=config.need_unitize)
