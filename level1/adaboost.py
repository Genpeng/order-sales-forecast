# _*_ coding: utf-8 _*_

"""
Use decision tree to predict the order amounts of next two month.

Author: Genpeng Xu (xgp1227@gmail.com)
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from data_loaders.level1_odi_data import Level1OdiData
from infers.level1_infer import Level1Infer


def main():
    ord_path = "../data/level2/m111-sku-order-all-final.csv"
    dis_path = None
    inv_path = None
    categories = ['CRZJ', 'CRXDG', 'CRYJ', 'JSJ', 'YSJ', 'RR', 'DR', 'CRXWJ']

    odi_data = Level1OdiData(ord_path, dis_path, inv_path, categories)
    X_train, y_train, X_test = odi_data.prepare_data_for_predict(4, left_bound='2016-04')

    estimator = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=5),
        n_estimators=800, random_state=np.random.RandomState(89)
    )
    infer = Level1Infer(estimator=estimator)
    preds_test, _ = infer.infer_future(X_train, y_train, X_test)


if __name__ == '__main__':
    main()
