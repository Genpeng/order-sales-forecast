# _*_ coding: utf-8 _*_

"""
Use decision tree to predict the order amounts of the next few months.

Author: Genpeng Xu
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from data_loader.level1_odi_data import Level1OdiData
from infer.level1_order_infer import Level1OrderInfer
from util.date_util import get_curr_date
from util.config_util import get_args, process_config


def main():
    # Load configuration
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

    print("[INFO] Parsing finished! ( ^ _ ^ ) V")
    print("[INFO] Selected model:", config.model)

    # xxx
    # ============================================================================================ #

    # categories = ['CRZJ', 'CRXDG', 'CRYJ', 'JSJ', 'YSJ', 'RR', 'DR', 'CRXWJ']
    categories = config.categories.split(',')
    start_pred_year, start_pred_month, start_pred_day = get_curr_date()
    # start_pred_year, start_pred_month, start_pred_day = 2019, 3, 10
    periods = 4 if start_pred_month + 4 <= 12 else 12 - start_pred_month

    print(categories)
    print(get_curr_date())

    # odi_data = Level1OdiData(start_pred_year, start_pred_month, start_pred_day, categories)
    # order, order_cate_month = odi_data.order, odi_data.level1_order
    # X_train, y_train, X_test = odi_data.prepare_data_for_predict(periods=4, left_bound='2016-04')
    #
    # estimator = AdaBoostRegressor(
    #     base_estimator=DecisionTreeRegressor(max_depth=5),
    #     n_estimators=800, random_state=np.random.RandomState(89)
    # )
    # infer = Level1OrderInfer()
    # preds_test, _ = infer.infer_future(X_train, y_train, X_test, periods)
    # months_pred = ['%d%02d' % (start_pred_year, m) for m in range(start_pred_month, start_pred_month + periods)]
    # df_pred_test = pd.DataFrame(
    #     np.array(preds_test).transpose(), index=order_cate_month.index,
    #     columns=months_pred
    # )
    # df_pred_test_more = infer.predict_more(order_cate_month, start_pred_year, start_pred_month, periods)
    # df_pred_test = pd.concat([df_pred_test, df_pred_test_more], axis=1)
    # df_pred_test = df_pred_test.stack().to_frame('pred_ord_qty')
    # df_pred_test.index.set_names(['first_cate_code', 'order_date'], inplace=True)
    # df_pred_test['pred_ord_qty'] = np.round(df_pred_test.pred_ord_qty, decimals=4)


if __name__ == '__main__':
    main()
