# _*_ coding: utf-8 _*_

"""
Update inventory forecast result of Level-3.

TODO: 线下常规和线下工程合并

Author: Genpeng Xu
"""

import time
import numpy as np
import pandas as pd
from bunch import Bunch

# Own customized modules
from data_loader.level3_inv_data_loader import Level3InvDataLoader
from infer.level3_infer import SalesInfer
from writer.kudu_result_writer import KuduResultWriter
from util.metric_util import add_accuracy
from util.config_util import get_args, process_config
from util.date_util import get_curr_date, infer_month, get_pre_months, timestamp_to_time


def update_history(pred_year, pred_month, model_config, db_config, table_name, batch_size=5000):
    """Update inventory forecast result of level3 in specified month."""
    print("\n[INFO] Current forecast month is: %d-%02d" % (pred_year, pred_month))

    # Step 1: Load & prepare training and testing set
    # ============================================================================================ #

    year_upper_bound, month_upper_bound = infer_month(pred_year, pred_month, offset=-2)
    train_months = get_pre_months(year_upper_bound, month_upper_bound, left_bound='2018-06')
    test_year, test_month = infer_month(pred_year, pred_month, offset=-1)
    print("[INFO] The last training month is: %d-%02d" % (year_upper_bound, month_upper_bound))
    print("[INFO] The test month is: %d-%02d" % (test_year, test_month))

    level3_inv_data = Level3InvDataLoader(pred_year, pred_month, config.categories, need_unitize=True)
    X_train, y_train = level3_inv_data.prepare_training_set(train_months, gap=1)
    X_test = level3_inv_data.prepare_testing_set(test_year, test_month, gap=1)

    # Step 2: Training and predicting
    # ============================================================================================ #

    level3_inv_infer = SalesInfer(config=model_config)
    pred_test, feat_imps = level3_inv_infer.predict(X_train, y_train, X_test)

    # Step 3: Process forecast result
    # ============================================================================================ #

    df_test = level3_inv_data.get_true_data(pred_year, pred_month)

    dt_pred = '%d%02d' % (pred_year, pred_month)
    df_pred_test = pd.DataFrame(
        np.array([pred_test]).transpose(), index=level3_inv_data.index, columns=[dt_pred]
    ).stack().to_frame('pred_inv_qty')
    df_pred_test.index.set_names(['customer_code', 'item_code', 'order_date'], inplace=True)
    df_pred_test['pred_inv_qty'] = df_pred_test.pred_inv_qty.apply(lambda x: x if x > 0 else 0)
    df_pred_test['pred_inv_qty'] = np.round(df_pred_test.pred_inv_qty, decimals=4)

    result = df_test.join(df_pred_test, how='left').reset_index()

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    customer_info = level3_inv_data.customer_info.to_dict()
    result['sales_cen_code'] = result.customer_code.map(customer_info['sales_cen_code'])
    result['sales_cen_name'] = result.customer_code.map(customer_info['sales_cen_name'])

    result['customer_name'] = result.customer_code.map(customer_info['customer_name'])
    # result['province_code'] = result.customer_code.map(customer_info['province_id'])
    # result['city_code'] = result.customer_code.map(customer_info['city_id'])
    # result['district_code'] = result.customer_code.map(customer_info['district_id'])
    # result['channel_code'] = result.customer_code.map(customer_info['channel_name_id'])
    result['province_name'] = result.customer_code.map(customer_info['province'])
    result['city_name'] = result.customer_code.map(customer_info['city'])
    result['district_name'] = result.customer_code.map(customer_info['district'])
    result['channel_name'] = result.customer_code.map(customer_info['channel_name'])
    result['platform_code'] = 'Pxxx'
    result['platform_name'] = '未知'

    sku_info = level3_inv_data.sku_info.to_dict()
    result['item_name'] = result.item_code.map(sku_info['item_name'])
    result['first_cate_code'] = result.item_code.map(sku_info['first_cate_code'])
    result['second_cate_code'] = result.item_code.map(sku_info['second_cate_code'])
    result['first_cate_name'] = result.item_code.map(sku_info['first_cate_name'])
    result['second_cate_name'] = result.item_code.map(sku_info['second_cate_name'])
    result['item_price'] = result.item_code.map(sku_info['item_price'])

    result['pred_inv_amount'] = np.round(result.pred_inv_qty * result.item_price, decimals=4)
    result['act_inv_amount'] = np.round(result.act_inv_qty * result.item_price, decimals=4)
    result['inv_pred_time'] = timestamp_to_time(time.time())

    add_accuracy(result, 'act_inv_qty', 'pred_inv_qty')

    result.rename(columns={'channel_name': 'channal_name'}, inplace=True)  # TODO: to be removed

    result['sales_cen_code'] = result.sales_cen_code.astype(str)
    result['customer_code'] = result.customer_code.astype(str)
    result['item_code'] = result.item_code.astype(str)

    # Step 4: Write into database (Kudu)
    # ============================================================================================ #

    level3_inv_writer = KuduResultWriter(db_config)
    level3_inv_writer.upsert(result, table_name, batch_size)


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

    if config.task_type == 'recent':
        curr_year, curr_month, _ = get_curr_date()
        pred_months = ['%d-%02d' % infer_month(curr_year, curr_month, offset=-1)]
    elif config.task_type == 'specified':
        pred_months = config.pred_months
    else:
        raise Exception("[ERROR] The task type is illegal! Please check the configuration file.")

    model_config = Bunch(config.model_config)
    db_config = Bunch(config.uat_db_config)

    for dt_str in pred_months:
        pred_year, pred_month = dt_str.split('-')
        update_history(int(pred_year), int(pred_month),
                       model_config, db_config,
                       config.table_name, config.batch_size)
