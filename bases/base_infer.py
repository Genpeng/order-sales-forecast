# _*_ coding: utf-8 _*_

"""
BaseInfer class, which is the superclass of all the infer class,
so all the infer class should inherit it before creating.

Author: Genpeng Xu
"""


class BaseInfer:
    def __init__(self, estimator):
        self._estimator = estimator

    def infer_future(self, X_train, y_train, X_test, periods):
        raise NotImplementedError
