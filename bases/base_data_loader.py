# _*_ coding: utf-8 _*_

"""
BaseDataLoader class, which is the superclass of all the data loader class,
so all the data loader class should inherit it before creating.

Author: Genpeng Xu
"""


class BaseDataLoader:
    def prepare_data_for_predict(self, periods, left_bound):
        raise NotImplementedError
