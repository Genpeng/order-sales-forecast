# _*_ coding: utf-8 _*_

"""
Level1 infer class.

Author: Genpeng Xu
"""

import time
import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import mean_squared_error

# Own customized modules
from bases.base_infer import BaseInfer
from util.metric_util import acc, mean_absolute_percent_error


class Level1OrderInfer(BaseInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer_future(self, X_train, y_train, X_test, periods=4):
        print("[INFO] Start training and predicting...")
        t0 = time.time()

        preds_train, preds_test = [], []
        for i in range(periods):
            print()
            print('# ' + '=' * 100 + ' #')
            print('# ' + 'Step %d' % i + ' ' * (100 - len('Step %d' % i)) + ' #')
            print('# ' + '=' * 100 + ' #')

            # Add previous predictions as a new feature
            if preds_train:
                X_train['m%s' % (i - 1)] = pd.Series(preds_train[i - 1])
                X_test['m%s' % (i - 1)] = pd.Series(preds_test[i - 1])

            # Adjust the month predicted
            if i != 0:
                X_train['pred_month'] = X_train.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)
                X_test['pred_month'] = X_test.pred_month.apply(lambda x: x - 11 if x + 1 > 12 else x + 1)

            print("[INFO] Fit the model...")
            self._estimator.fit(X_train.values, y_train[:, i])

            # Predict
            pred_train = self._estimator.predict(X_train.values)
            pred_test = self._estimator.predict(X_test.values)

            # Calculate accuracy
            acc_train = acc(y_train[:, i], pred_train)
            print("[INFO] The accuracy of training set is: %.2f%%" % (acc_train * 100))

            # Calculate MAPE
            mape_train = mean_absolute_percent_error(y_train[:, i], pred_train)
            print("[INFO] The MAPE of training set is: %.4f" % (mape_train))

            # Calculate MSE
            mse_train = mean_squared_error(y_train[:, i], pred_train)
            print("[INFO] The MSE of training set is: %.4f" % (mse_train))

            # Store the intermediate results
            preds_train.append(pred_train)
            preds_test.append(pred_test)

            # Output feature importances
            feat_imps = sorted(zip(X_train.columns, self._estimator.feature_importances_),
                               key=lambda x: x[1], reverse=True)
            print("The feature importances are as follow: ")
            print('\n'.join('%s: %s' % (feat_name, feat_imp) for feat_name, feat_imp in feat_imps))

        print()
        print("[INFO] Finished! ( ^ _ ^ ) V")
        print("[INFO] Done in %f seconds." % (time.time() - t0))

        return preds_test, feat_imps

    def predict_more(self, order_cate_month, start_pred_year, start_pred_month, periods=4):
        if start_pred_month + periods > 12:
            return None
        else:
            start_aver_month = start_pred_month + periods
        history = []
        for y in np.arange(start_pred_year - 1, 2015, -1):
            start_dt, end_dt = date(y, start_aver_month, 1), date(y + 1, 1, 1)
            tmp = order_cate_month[pd.date_range(start_dt, end_dt, freq='M')].values
            history.append(tmp)
        result = np.mean(np.array(history), axis=0)
        months_pred = ['%d%02d' % (start_pred_year, m) for m in range(start_aver_month, 13)]
        result = pd.DataFrame(result,
                              index=order_cate_month.index,
                              columns=months_pred)
        return result
