# _*_ coding: utf-8 _*_

"""
BaseInfer class, which is the superclass of all the infer class,
so all the infer class should inherit it before creating.

Author: Genpeng Xu
"""

import numpy as np
import pandas as pd
from bunch import Bunch
from typing import Union
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm.sklearn import LGBMRegressor


class BaseInfer(object):
    def __init__(self, config: Bunch) -> None:
        self._config = config
        self._model_params = self._config.model_params
        self._estimator = self._get_estimator()

    def _get_estimator(self):
        if self._config.model == 'decision_tree':
            estimator = DecisionTreeRegressor(**self._model_params)
        elif self._config.model == 'random_forest':
            estimator = RandomForestRegressor(**self._model_params)
        elif self._config.model == 'gradient_boosting':
            estimator = GradientBoostingRegressor(**self._model_params)
        elif self._config.model == 'adaboost':
            self._model_params['base_estimator'] = DecisionTreeRegressor(**self._config.base_estimator_params)
            estimator = AdaBoostRegressor(**self._model_params)
        elif self._config.model == 'lightgbm':
            estimator = LGBMRegressor(**self._model_params)
        else:
            raise Exception("[ERROR] Please check the model name!")
        return estimator

    def fit(self, X_train: pd.DataFrame, y_train: Union[pd.DataFrame, np.ndarray]) -> None:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @property
    def config(self):
        return self._config

    @property
    def model_params(self):
        return self._model_params
