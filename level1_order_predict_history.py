# _*_ coding: utf-8 _*_

"""
Update order forecast result of Level-2.

Author: Genpeng Xu
"""

import time
import numpy as np
from bunch import Bunch
from datetime import datetime, date

# Own customized modules
from infer.sales_infer import LGBMSalesInfer
from util.metric_util import add_accuracy
from global_vars import SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG
from util.config_util import get_args, process_config
from util.feature_util import rule_func
from data_loader.item_list import ItemList
from data_loader.level2_data import Level2DataLoader
from data_loader.plan_data import PlanData
from writer.kudu_result_writer import KuduResultWriter
from util.date_util import get_curr_date, infer_month, get_pre_months, timestamp_to_time


def update_history_for_level1_order(level2_data: Level2DataLoader,
                                    plan_data: PlanData,
                                    model_config: Bunch,
                                    db_config: Bunch,
                                    start_pred_year: int,
                                    start_pred_month: int,
                                    gap: int,
                                    use_unitize: bool = True) -> None:
    """Update order forecast result of level2 in specified month."""

    # Step 1: Prepare training and testing set
    # ============================================================================================ #

    last_train_year, last_train_month = infer_month(start_pred_year, start_pred_month, offset=-gap)
    train_months = get_pre_months(last_train_year, last_train_month, left_bound='2016-03')
    true_pred_year, true_pred_month = infer_month(start_pred_year, start_pred_month, gap)

    X_train, y_train = level2_data.prepare_training_set(train_months, gap=gap)
    X_test = level2_data.prepare_testing_set(start_pred_year, start_pred_month, gap=gap)

    # Step 2: Training and predicting
    # ============================================================================================ #

    level2_order_infer = LGBMSalesInfer(model_config)
    level2_order_infer.fit(X_train, y_train)
    preds_test = level2_order_infer.predict(X_test)

    # Step 3: Process forecast result
    # ============================================================================================ #

    df_test = level2_data.get_true_order_data(true_pred_year, true_pred_month)
    df_pred_test = level2_data.decorate_pred_result(preds_test,
                                                    true_pred_year,
                                                    true_pred_month,
                                                    use_unitize=use_unitize)

    result = df_test.join(df_pred_test, how='left').reset_index()

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

    result['act_ord_amount'] = np.round(result.act_ord_qty * result.item_price, decimals=4 if use_unitize else 0)
    result['pred_ord_amount'] = np.round(result.pred_ord_qty * result.item_price, decimals=4 if use_unitize else 0)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    add_accuracy(result, 'ord_acc', 'act_ord_qty', 'pred_ord_qty')
    result['ord_weighted_acc'] = (result.act_ord_qty * result.ord_acc).astype(np.float32)

    item_list = ItemList(start_pred_year, start_pred_month)
    result = result.loc[result.item_code.apply(lambda x: item_list.is_white_items(x))]

    print()
    print("[INFO] The average accuracy of model is: %.2f" % (result.ord_acc.mean() * 100))
    print("[INFO] The weighted accuracy of model is: %.2f" % (result.ord_weighted_acc.sum() / result.act_ord_qty.sum() * 100))

    # Step 4: Ensemble with rule
    # ============================================================================================ #

    rule_res = result.copy()
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
    new_items_by_dis = set(dis_sku_month_pre3.loc[(dis_sku_month_pre3.num_not_null == 1) & (dis_sku_month_pre3.iloc[:, 2] > 0)].index)

    demand = plan_data.get_one_month(true_pred_year, true_pred_month, True)
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

    add_accuracy(rule_res, 'rule_ord_acc', 'act_ord_qty', 'pred_ord_qty_rule')
    rule_res['rule_ord_weighted_acc'] = (rule_res.act_ord_qty * rule_res.rule_ord_acc).astype(np.float32)

    print()
    print("[INFO] The average accuracy of rule is: %.2f" % (rule_res.rule_ord_acc.mean() * 100))
    print("[INFO] The weighted accuracy of rule is: %.2f" % (rule_res.rule_ord_weighted_acc.sum() / rule_res.act_ord_qty.sum() * 100))

    result['pred_ord_qty'] = result.pred_ord_qty * 0.5 + rule_res.pred_ord_qty_rule * 0.5

    add_accuracy(result, 'ord_acc', 'act_ord_qty', 'pred_ord_qty')
    result['ord_weighted_acc'] = (result.act_ord_qty * result.ord_acc).astype(np.float32)

    print()
    print("[INFO] The average accuracy of ensemble is: %.2f" % (result.ord_acc.mean() * 100))
    print("[INFO] The weighted accuracy of ensemble is: %.2f" % (result.ord_weighted_acc.sum() / result.act_ord_qty.sum() * 100))

    # Step 5: Write into database (Kudu)
    # ============================================================================================ #

    if db_config.env == 'SIT':
        level2_order_writer = KuduResultWriter(Bunch(SIT_DB_CONFIG))
    elif db_config.env == 'UAT':
        level2_order_writer = KuduResultWriter(Bunch(UAT_DB_CONFIG))
    elif db_config.env == 'PROD':
        level2_order_writer = KuduResultWriter(Bunch(PROD_DB_CONFIG))
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")
    level2_order_writer.clear_one_month(db_config.table_name, 'order_date', true_pred_year, true_pred_month)
    level2_order_writer.upsert(result, db_config.table_name, db_config.batch_size)


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

    # Update forecast result of level2 order
    # ============================================================================================ #

    curr_year, curr_month, _ = get_curr_date()
    gap = 1  # 更新历史，默认预测M1月
    year_upper_bound, month_upper_bound = infer_month(curr_year, curr_month, offset=-(gap+1))

    if config.task_type == 'recent':
        pred_months = ['%d-%02d' % (year_upper_bound, month_upper_bound)]
    elif config.task_type == 'specified':
        pred_months = config.pred_months
    else:
        raise Exception("[ERROR] The task type is illegal! Please check the configuration file.")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.db_config)
    level2_data = Level2DataLoader(curr_year, curr_month,
                                   categories=config.categories,
                                   need_unitize=config.need_unitize,
                                   label_data='order')
    if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
        p_y, p_m = infer_month(curr_year, curr_month, -1)
        plan_data = PlanData(p_y, p_m, need_unitize=config.need_unitize)
    else:
        plan_data = PlanData(curr_year, curr_month, need_unitize=config.need_unitize)

    for ym_str in pred_months:
        start_pred_year, start_pred_month = map(int, ym_str.split('-'))
        if date(start_pred_year, start_pred_month, 1) <= date(year_upper_bound, month_upper_bound, 1):
            update_history_for_level1_order(level2_data=level2_data,
                                            plan_data=plan_data,
                                            model_config=model_config,
                                            db_config=db_config,
                                            start_pred_year=start_pred_year,
                                            start_pred_month=start_pred_month,
                                            gap=gap,
                                            use_unitize=config.need_unitize)
        else:
            raise Exception("[INFO] The update date is illegal!!!")
