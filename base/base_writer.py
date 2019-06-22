# _*_ coding: utf-8 _*_

"""
BaseWriter class, which is the superclass of all the writer class,
so all the writer class should inherit it before creating.

Author: Genpeng Xu
"""

from util.db_util import create_cursor


class BaseWriter:
    def __init__(self, config):
        self._config = config
        self._cursor = create_cursor(config)

    def upsert(self, result, table_name, batch_size=5000):
        raise NotImplementedError

