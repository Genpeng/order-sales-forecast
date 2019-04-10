# _*_ coding: utf-8 _*_

"""
BaseInfer class, which is the superclass of all the infer class,
so all the infer class should inherit it before creating.

Author: Genpeng Xu
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


class BaseInfer:
    def __init__(self, config):
        self._config = config
        self._estimator = self._get_estimator(self._config)

    def _get_estimator(self, config):
        rng = np.random.RandomState(89)
        if config.model == 'decision_tree':
            estimator = DecisionTreeRegressor(max_depth=config.max_depth, random_state=rng)
        elif config.model == 'random_forest':
            estimator = RandomForestRegressor(n_estimators=config.n_estimators,
                                              max_depth=config.max_depth,
                                              n_jobs=config.n_jobs,
                                              random_state=rng)
        elif config.model == 'gradient_boosting':
            estimator = GradientBoostingRegressor(learning_rate=config.learning_rate,
                                                  n_estimators=config.n_estimators,
                                                  max_depth=config.max_depth,
                                                  random_state=rng)
        elif config.model == 'adaboost':
            estimator = AdaBoostRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=config.max_depth, random_state=rng),
                n_estimators=config.n_estimator,
                learning_rate=config.learning_rate,
                random_state=rng
            )
        else:
            raise Exception("[ERROR] Please check the model name!")
        return estimator

    def infer_future(self, X_train, y_train, X_test, periods):
        raise NotImplementedError
