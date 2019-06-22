# _*_ coding: utf-8 _*_

"""
Write inventory forecast result of Level-3.

Author: Genpeng Xu
"""

from util.db_util import batch_upsert_data
from base.base_writer import BaseWriter


class Level3InvWriter(BaseWriter):
    """Inventory forecast result writer of level3."""

    def upsert(self, result, table_name, batch_size=5000):
        batch_upsert_data(self._cursor, result, table_name, batch_size)


def main():
    from global_vars import UatDbConfig
    db_config = UatDbConfig()
    writer = Level3InvWriter(db_config)


if __name__ == '__main__':
    main()
