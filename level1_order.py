# _*_ coding: utf-8 _*_

"""
Predict the order amounts of the next few months by using machine learning or
deep learning algorithms.

Author: Genpeng Xu
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

# Own customized modules
from data_loader.level1_odi_data import Level1OdiData
from infer.level1_order_infer import Level1OrderInfer
from util.date_util import get_curr_date, timestamp_to_time
from util.config_util import get_args, process_config

warnings.filterwarnings('ignore')


def main():
    # Step 1: Load configuration
    # ============================================================================================ #

    print("[INFO] Load configuration...")

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
    print("[INFO] Selected model:", config.model)

    # Step 2: Train and predict
    # ============================================================================================ #

    categories = config.categories.split(',')
    # start_pred_year, start_pred_month, start_pred_day = get_curr_date()
    start_pred_year, start_pred_month, start_pred_day = 2019, 3, 10  # TODO: will be removed
    periods = 4 if start_pred_month + 4 <= 12 else 12 - start_pred_month

    # prepare data
    odi_data = Level1OdiData(start_pred_year, start_pred_month, start_pred_day, categories)
    order, order_cate_month, cate_aver_price = odi_data.order, odi_data.level1_order, odi_data.cate_aver_price
    X_train, y_train, X_test = odi_data.prepare_data_for_predict(periods=4, left_bound='2016-04')

    # create predictor and use it to train and predict
    infer = Level1OrderInfer(config=config)
    preds_test, _ = infer.infer_future(X_train, y_train, X_test, periods=periods)  # output is list of ndarray
    months_pred = ['%d%02d' % (start_pred_year, m) for m in range(start_pred_month, start_pred_month + periods)]
    df_pred_test = pd.DataFrame(
        np.array(preds_test).transpose(), index=order_cate_month.index,
        columns=months_pred
    )
    df_pred_test_more = infer.predict_more(order_cate_month,
                                           start_pred_year,
                                           start_pred_month,
                                           periods=periods)  # output is DataFrame object

    # Step 3: Save result
    # ============================================================================================ #

    print("[INFO] Start saving result...")

    if df_pred_test_more is not None:
        df_pred_test = pd.concat([df_pred_test, df_pred_test_more], axis=1)
    df_pred_test = df_pred_test.stack().to_frame('pred_ord_qty')
    df_pred_test.index.set_names(['first_cate_code', 'order_date'], inplace=True)
    df_pred_test['pred_ord_qty'] = np.round(df_pred_test.pred_ord_qty, decimals=4)

    # calculate actual order quantity of 'M' month
    start_dt_str = '%d-%02d-%02d' % (start_pred_year, start_pred_month, 1)
    end_dt_str = '%d-%02d-%02d' % (start_pred_year, start_pred_month, start_pred_day)
    order_m = order.loc[(order.order_date >= start_dt_str) & (order.order_date <= end_dt_str)]
    order_m['order_date'] = '%d-%02d' % (start_pred_year, start_pred_month)
    order_m = order_m.groupby(['category', 'order_date'])[['ord_qty']].sum()
    order_m.rename(columns={'ord_qty': 'act_ord_qty'}, inplace=True)
    order_m.index.set_names(['first_cate_code', 'order_date'], inplace=True)

    # add other necessary fields
    cate_code_to_name = {'CRYJ': '烟机',
                         'CRZJ': '灶具',
                         'CRXDG': '消毒柜',
                         'CRXWJ': '洗碗机',
                         'DR': '电热水器',
                         'RR': '燃气热水器',
                         'JSJ': '净水机',
                         'YSJ': '饮水机'}
    result = df_pred_test.join(order_m, how='left').fillna(0)
    result.reset_index(inplace=True)
    result['first_cate_name'] = result.first_cate_code.map(cate_code_to_name)
    result['comb_name'] = 'Default'
    result['pred_time'] = timestamp_to_time(time.time())
    result['aver_price'] = result.first_cate_code.map(cate_aver_price.to_dict()['price'])
    result['pred_ord_amount'] = np.round(result.pred_ord_qty * result.aver_price, decimals=4)
    result['act_ord_amount'] = np.round(result.act_ord_qty * result.aver_price, decimals=4)
    result['accuracy'] = 0
    result['bu_code'] = 'M111'
    result['bu_name'] = '厨房热水器事业部'

    # modify column order
    result = result[['bu_code',
                     'comb_name',
                     'order_date',
                     'first_cate_code',
                     'bu_name',
                     'first_cate_name',
                     'act_ord_qty',
                     'pred_ord_qty',
                     'accuracy',
                     'act_ord_amount',
                     'pred_ord_amount',
                     'aver_price',
                     'pred_time']]

    # save to file
    result_root_dir = "results"
    if not os.path.exists(result_root_dir):
        os.mkdir(result_root_dir)
    filename = "level1_order_%s_%d%02d%02d_%d.txt" % (config.model,
                                                      start_pred_year,
                                                      start_pred_month,
                                                      start_pred_day,
                                                      time.time())
    result_path = os.path.join(result_root_dir, filename)
    result.to_csv(result_path, sep=',', index=None)

    print("[INFO] Saving finished!")


if __name__ == '__main__':
    main()
