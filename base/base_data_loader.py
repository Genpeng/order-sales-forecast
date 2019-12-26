# _*_ coding: utf-8 _*_

"""
BaseDataLoader class, which is the superclass of all the data loader class,
so all the data loader class should inherit it before creating.

Author: Genpeng Xu
"""


class BaseDataLoader:
    def prepare_training_set(self, months, gap=0):
        raise NotImplementedError

    def prepare_val_set(self, pred_year, pred_month, gap=0):
        raise NotImplementedError

    def prepare_testing_set(self, pred_year, pred_month, gap=0):
        raise NotImplementedError
