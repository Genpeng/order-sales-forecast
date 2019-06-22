# _*_ coding: utf-8 _*_

"""
Write inventory forecast result of Level-3.

Author: Genpeng Xu
"""

from util.db_util import batch_upsert_data
from base.base_writer import BaseWriter


class Level3InvWriter(BaseWriter):
    """Inventory forecast result writer of level3."""

    def __init__(self, config):
        super().__init__(config)

    def clear(self, table_name):
        self._cursor.execute("DELETE FROM %s" % table_name)

    def upsert(self, result, table_name, batch_size=5000):
        batch_upsert_data(self._cursor, result, table_name, batch_size)


def main():
    from global_vars import UatDbConfig

    db_config = UatDbConfig()
    writer = Level3InvWriter(db_config)
    writer.clear("level3_customer_pred_result")


if __name__ == '__main__':
    main()
