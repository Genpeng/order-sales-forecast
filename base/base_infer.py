# _*_ coding: utf-8 _*_

"""
BaseInfer class, which is the superclass of all the infer class,
so all the infer class should inherit it before creating.

Author: Genpeng Xu
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm.sklearn import LGBMRegressor


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
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=rng
            )
        elif config.model == 'lightgbm':
            estimator = LGBMRegressor(num_leaves=config.num_leaves,
                                      learning_rate=config.learning_rate,
                                      objective=config.objective,
                                      random_state=config.random_state,
                                      n_jobs=config.n_jobs,
                                      # key word parameters
                                      num_iterations=config.num_iterations,
                                      min_data_in_leaf=config.min_data_in_leaf,
                                      feature_fraction=config.feature_fraction,
                                      bagging_fraction=config.bagging_fraction,
                                      bagging_freq=config.bagging_freq,
                                      metric=config.metric)
        else:
            raise Exception("[ERROR] Please check the model name!")
        return estimator

    def predict_future(self, X_train, y_train, X_test, periods):
        raise NotImplementedError
