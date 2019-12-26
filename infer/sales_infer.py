# _*_ coding: utf-8 _*_

"""
Sales predictor.

Author: Genpeng Xu
"""

import time
import numpy as np
import pandas as pd
from bunch import Bunch
from typing import Union
from sklearn.metrics import mean_squared_error

# Own customized modules
from base.base_infer import BaseInfer
from util.metric_util import mean_absolute_percent_error


class SalesInfer(BaseInfer):
    def __init__(self, config: Bunch) -> None:
        super().__init__(config)

    def fit(self, X_train: pd.DataFrame, y_train: Union[pd.DataFrame, np.ndarray]) -> None:
        print("[INFO] Start training...")
        t0 = time.time()
        self._estimator.fit(X_train, y_train,
                            eval_set=[(X_train, y_train)],
                            eval_names=['train'],
                            verbose=True)

        pred_train = self.predict(X_train)

        # Calculate MAPE
        mape_train = mean_absolute_percent_error(y_train, pred_train)
        print("[INFO] mape_train: %.4f" % mape_train)

        # Calculate RMSE
        rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
        print("[INFO] rmse_train: %.4f" % rmse_train)

        # Feature importances
        feat_imps = sorted(zip(X_train.columns, self._estimator.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        print("\nThe feature importances are as follow: ")
        print('\n'.join('%s: %s' % (feat_name, feat_imp) for feat_name, feat_imp in feat_imps))

        print()
        print("[INFO] Training finished! ( ^ _ ^ ) V")
        print("[INFO] Done in %f seconds." % (time.time() - t0))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_iter = self._estimator.best_iteration_ or self._model_params['num_iterations']
        return self._estimator.predict(X, num_iteration=n_iter)


def _test():
    # prepare data
    from data_loader.level3_order_data import Level3OrderDataLoader
    curr_year, curr_month, _ = 2019, 11, 2
    train_months = ['2019-09', '2019-08']
    level3_order_data = Level3OrderDataLoader(curr_year, curr_month)
    X_train, y_train = level3_order_data.prepare_training_set(train_months, gap=0)
    X_val, y_val = level3_order_data.prepare_val_set(2019, 10, gap=0)

    # create a predictor, train and predict
    from bunch import Bunch
    from util.metric_util import acc_v2
    model_config = {
        "model": "lightgbm",
        "model_params": {
            "num_leaves": 80,
            "learning_rate": 0.01,
            "objective": "regression",
            "random_state": 89,
            "n_jobs": 16,
            "num_iterations": 500,
            "early_stopping_round": 150,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "metric": "l2"
        }
    }
    model_config = Bunch(model_config)
    level3_order_infer = SalesInfer(model_config)
    level3_order_infer.fit(X_train, y_train)
    preds_val = level3_order_infer.predict(X_val)
    print(acc_v2(y_val, preds_val))


if __name__ == '__main__':
    _test()
