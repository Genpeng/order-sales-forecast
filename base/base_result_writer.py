# _*_ coding: utf-8 _*_

"""
BaseResultWriter class, which is the superclass of all the result writer class,
so all the result writer class should inherit it before creating.

Author: Genpeng Xu
"""


class BaseResultWriter:
    def __init__(self, config):
        self._config = config

    def clear(self, table_name):
        raise NotImplementedError

    def upsert(self, result, table_name, batch_size=5000):
        raise NotImplementedError
