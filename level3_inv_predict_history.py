# _*_ coding: utf-8 _*_

"""
Update inventory forecast result of Level-3.

Author: Genpeng Xu
"""

import time
import numpy as np
from bunch import Bunch
from datetime import date

# Own customized modules
from infer.sales_infer import LGBMSalesInfer
from util.metric_util import add_accuracy
from global_vars import SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG
from util.config_util import get_args, process_config
from data_loader.item_list import ItemList
from data_loader.customer_list import CustomerList
from data_loader.level3_inv_data import Level3InvDataLoader
from writer.kudu_result_writer import KuduResultWriter
from util.date_util import get_curr_date, infer_month, get_pre_months, timestamp_to_time


def update_history_for_level3_inv(level3_inv_data: Level3InvDataLoader,
                                  model_config: Bunch,
                                  db_config: Bunch,
                                  start_pred_year: int,
                                  start_pred_month: int,
                                  gap: int,
                                  use_unitize: bool = True) -> None:
    """Update inventory forecast result of level3 in specified month."""

    # Step 1: Prepare training and testing set
    # ============================================================================================ #

    last_train_year, last_train_month = infer_month(start_pred_year, start_pred_month, offset=-gap)
    train_months = get_pre_months(last_train_year, last_train_month, left_bound='2018-06')
    true_pred_year, true_pred_month = infer_month(start_pred_year, start_pred_month, gap)

    X_train, y_train = level3_inv_data.prepare_training_set(train_months, gap=gap)
    X_test = level3_inv_data.prepare_testing_set(start_pred_year, start_pred_month, gap=gap)

    # Step 2: Training and predicting
    # ============================================================================================ #

    level3_inv_infer = LGBMSalesInfer(model_config)
    level3_inv_infer.fit(X_train, y_train)
    preds_test = level3_inv_infer.predict(X_test)

    # Step 3: Process forecast result
    # ============================================================================================ #

    df_test = level3_inv_data.get_true_data(true_pred_year, true_pred_month)
    df_preds_test = level3_inv_data.decorate_pred_result(preds_test,
                                                         true_pred_year,
                                                         true_pred_month,
                                                         use_unitize=use_unitize)

    result = df_test.join(df_preds_test, how='left').reset_index()

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    customer_info_dict = level3_inv_data.customer_info.to_dict()
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

    sku_info_dict = level3_inv_data.sku_info.to_dict()
    result['item_name'] = result.item_code.map(sku_info_dict['item_name'])
    result['first_cate_code'] = result.item_code.map(sku_info_dict['first_cate_code'])
    result['second_cate_code'] = result.item_code.map(sku_info_dict['second_cate_code'])
    result['first_cate_name'] = result.item_code.map(sku_info_dict['first_cate_name'])
    result['second_cate_name'] = result.item_code.map(sku_info_dict['second_cate_name'])
    result['channel_name'] = result.item_code.map(sku_info_dict['channel_name'])
    result['item_price'] = result.item_code.map(sku_info_dict['item_price'])

    result['act_inv_amount'] = np.round(result.act_inv_qty * result.item_price, decimals=4 if use_unitize else 0)
    result['pred_inv_amount'] = np.round(result.pred_inv_qty * result.item_price, decimals=4 if use_unitize else 0)
    result['inv_pred_time'] = timestamp_to_time(time.time())

    add_accuracy(result, 'inv_acc', 'act_inv_qty', 'pred_inv_qty')
    result['inv_weighted_acc'] = (result.act_inv_qty * result.inv_acc).astype(np.float32)

    customer_list = CustomerList()
    item_list = ItemList(start_pred_year, start_pred_month)
    result = result.loc[result.customer_code.apply(lambda x: customer_list.is_white_customer(x))]
    result = result.loc[result.item_code.apply(lambda x: item_list.is_white_items(x))]

    print()
    print("[INFO] The average accuracy is: %.2f" % (result.inv_acc.mean() * 100))
    print("[INFO] The weighted accuracy is: %.2f" % (result.inv_weighted_acc.sum() / result.act_inv_qty.sum() * 100))

    # Step 4: Write into database (Kudu)
    # ============================================================================================ #

    if db_config.env == 'SIT':
        level3_order_writer = KuduResultWriter(Bunch(SIT_DB_CONFIG))
    elif db_config.env == 'UAT':
        level3_order_writer = KuduResultWriter(Bunch(UAT_DB_CONFIG))
    elif db_config.env == 'PROD':
        level3_order_writer = KuduResultWriter(Bunch(PROD_DB_CONFIG))
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")
    level3_order_writer.clear_one_month(db_config.table_name, 'order_date', true_pred_year, true_pred_month)
    level3_order_writer.upsert(result, db_config.table_name, db_config.batch_size)


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

    # Update forecast result of level3 inventory
    # ============================================================================================ #

    curr_year, curr_month, _ = get_curr_date()
    gap = 1  # 更新历史，默认预测M1月
    year_upper_bound, month_upper_bound = infer_month(curr_year, curr_month, offset=-(gap + 1))

    if config.task_type == 'recent':
        pred_months = ['%d-%02d' % (year_upper_bound, month_upper_bound)]
    elif config.task_type == 'specified':
        pred_months = config.pred_months
    else:
        raise Exception("[ERROR] The task type is illegal! Please check the configuration file.")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.db_config)
    level3_inv_data = Level3InvDataLoader(curr_year, curr_month,
                                          categories=config.categories, need_unitize=config.need_unitize)

    for ym_str in pred_months:
        start_pred_year, start_pred_month = map(int, ym_str.split('-'))
        if date(start_pred_year, start_pred_month, 1) <= date(year_upper_bound, month_upper_bound, 1):
            update_history_for_level3_inv(level3_inv_data=level3_inv_data,
                                          model_config=model_config,
                                          db_config=db_config,
                                          start_pred_year=start_pred_year,
                                          start_pred_month=start_pred_month,
                                          gap=gap,
                                          use_unitize=config.need_unitize)
        else:
            raise Exception("[INFO] The update date is illegal!!!")
