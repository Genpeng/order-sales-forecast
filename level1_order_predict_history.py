# _*_ coding: utf-8 _*_

"""
Update order forecast result of Level-2.

Author: Genpeng Xu
"""

import time
import warnings
import numpy as np
from bunch import Bunch
from datetime import date

# Own customized modules
from infer.sales_infer import RFSalesInfer
from util.metric_util import add_accuracy
from global_vars import SIT_DB_CONFIG, UAT_DB_CONFIG, PROD_DB_CONFIG
from util.config_util import get_args, process_config
from data_loader.level1_data import Level1DataLoader
from writer.kudu_result_writer import KuduResultWriter
from util.date_util import get_curr_date, infer_month, get_pre_months, timestamp_to_time

warnings.filterwarnings('ignore')


def update_history_for_level1_order(level1_data: Level1DataLoader,
                                    model_config: Bunch,
                                    db_config: Bunch,
                                    start_pred_year: int,
                                    start_pred_month: int,
                                    gap: int,
                                    use_unitize: bool = True) -> None:
    """Update order forecast result of level1 in specified month."""

    # Step 1: Prepare training and testing set
    # ============================================================================================ #

    last_train_year, last_train_month = infer_month(start_pred_year, start_pred_month, offset=-gap)
    train_months = get_pre_months(last_train_year, last_train_month, left_bound='2016-03')
    true_pred_year, true_pred_month = infer_month(start_pred_year, start_pred_month, gap)

    X_train, y_train = level1_data.prepare_training_set(train_months, gap=gap)
    X_test = level1_data.prepare_testing_set(start_pred_year, start_pred_month, gap=gap)

    # Step 2: Training and predicting
    # ============================================================================================ #

    level1_infer = RFSalesInfer(model_config)
    level1_infer.fit(X_train, y_train)
    preds_test = level1_infer.predict(X_test)

    # Step 3: Process forecast result
    # ============================================================================================ #

    df_test = level1_data.get_true_order_data(true_pred_year, true_pred_month)
    df_pred_test = level1_data.decorate_pred_result(preds_test,
                                                    true_pred_year,
                                                    true_pred_month,
                                                    use_unitize=use_unitize)

    result = df_test.join(df_pred_test, how='left').reset_index()

    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'
    result['comb_name'] = 'Default'

    cate_code_to_name = {
        'CRYJ': '烟机',
        'CRZJ': '灶具',
        'CRXDG': '消毒柜',
        'CRXWJ': '洗碗机',
        'DR': '电热水器',
        'RR': '燃气热水器',
        'JSJ': '净水机',
        'YSJ': '饮水机'
    }
    result['first_cate_name'] = result.first_cate_code.map(cate_code_to_name)
    cate_info_dict = level1_data.cate_info.to_dict()
    result['aver_price'] = result.first_cate_code.map(cate_info_dict['cate_aver_price'])

    result['act_ord_amount'] = np.round(result.act_ord_qty * result.aver_price, decimals=4 if use_unitize else 0)
    result['pred_ord_amount'] = np.round(result.pred_ord_qty * result.aver_price, decimals=4 if use_unitize else 0)
    result['ord_pred_time'] = timestamp_to_time(time.time())

    add_accuracy(result, 'ord_acc', 'act_ord_qty', 'pred_ord_qty')
    result['ord_weighted_acc'] = (result.act_ord_qty * result.ord_acc).astype(np.float32)

    print()
    print("[INFO] The average accuracy of model is: %.2f" % (result.ord_acc.mean() * 100))
    print("[INFO] The weighted accuracy of model is: %.2f" % (
            result.ord_weighted_acc.sum() / result.act_ord_qty.sum() * 100))

    # Step 4: Write into database (Kudu)
    # ============================================================================================ #

    if db_config.env == 'SIT':
        kudu_config = SIT_DB_CONFIG
    elif db_config.env == 'UAT':
        kudu_config = UAT_DB_CONFIG
    elif db_config.env == 'PROD':
        kudu_config = PROD_DB_CONFIG
    else:
        raise Exception("[INFO] The environment name of database to write result is illegal!!!")
    level1_writer = KuduResultWriter(Bunch(kudu_config))
    level1_writer.clear_one_month(db_config.table_name, 'order_date', true_pred_year, true_pred_month)
    level1_writer.upsert(result, db_config.table_name, db_config.batch_size)


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

    # Update forecast result of level1 order
    # ============================================================================================ #

    # curr_year, curr_month, _ = get_curr_date()
    curr_year, curr_month, _ = 2019, 12, 16
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
    level1_data = Level1DataLoader(curr_year, curr_month,
                                   categories=config.categories,
                                   need_unitize=config.need_unitize,
                                   label_data='order')

    for ym_str in pred_months:
        start_pred_year, start_pred_month = map(int, ym_str.split('-'))
        if date(start_pred_year, start_pred_month, 1) <= date(year_upper_bound, month_upper_bound, 1):
            update_history_for_level1_order(level1_data=level1_data,
                                            model_config=model_config,
                                            db_config=db_config,
                                            start_pred_year=start_pred_year,
                                            start_pred_month=start_pred_month,
                                            gap=gap,
                                            use_unitize=config.need_unitize)
        else:
            raise Exception("[INFO] The update date is illegal!!!")
