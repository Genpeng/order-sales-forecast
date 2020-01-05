# _*_ coding: utf-8 _*_

"""
Write inventory forecast result of Level-3.

Author: Genpeng Xu
"""

import time
from impala.dbapi import connect
from base.base_result_writer import BaseResultWriter


def _create_cursor(config):
    conn = connect(host=config.host,
                   port=config.port,
                   database=config.database,
                   auth_mechanism=config.auth_mechanism,
                   user=config.user,
                   password=config.password)
    return conn.cursor()


def _clear_all(cursor, table_name):
    delete_sql = "DELETE FROM %s" % table_name
    cursor.execute(delete_sql)


def _clear_one_month(cursor, table_name, date_col, year, month):
    delete_sql = "DELETE FROM %s WHERE `%s` = '%s%s'" % (table_name, date_col, year, month)
    cursor.execute(delete_sql)


def _clear_one_period(cursor, table_name, date_col, start_year, start_month, end_year, end_month):
    delete_sql = "DELETE FROM %s WHERE `%s` >= '%s%s' and `%s` <= '%s%s'" % (table_name,
                                                                             date_col,
                                                                             start_year,
                                                                             start_month,
                                                                             date_col,
                                                                             end_year,
                                                                             end_month)
    cursor.execute(delete_sql)


def _clear_months_after(cursor, table_name, date_col, start_year, start_month):
    delete_sql = "DELETE FROM %s WHERE `%s` >= '%s%s'" % (table_name,
                                                          date_col,
                                                          start_year,
                                                          start_month)
    cursor.execute(delete_sql)


def _batch_upsert_data(cursor, df, table_name, batch_size=5000):
    print("[INFO] Start inserting or updating data...")
    t0 = time.time()

    l = len(df)  # the size of data
    n = (l - 1) // batch_size + 1  # number of times to write
    header_str = str(tuple(df.columns)).replace("\'", '')
    for i in range(n):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, l)
        values = [str(tuple(row)) for row in df[start_index:end_index].values]
        values_str = ', '.join(values)
        sql = "UPSERT INTO %s %s VALUES %s;" % (table_name, header_str, values_str)
        try:
            cursor.execute(sql)
        except Exception:
            print("[ERROR] Upsert failed!!!")
            return

    print("[INFO] Upsert finished! ( ^ _ ^ )V")
    print("[INFO] Done in %s seconds." % (time.time() - t0))


class KuduResultWriter(BaseResultWriter):
    """ResultWriter used to write result into Kudu database by Impala."""

    def __init__(self, config):
        super().__init__(config)
        self._cursor = _create_cursor(config)

    def clear(self, table_name):
        _clear_all(self._cursor, table_name)

    def clear_one_month(self, table_name, date_col, year, month):
        _clear_one_month(self._cursor, table_name, date_col, year, month)

    def clear_one_period(self, table_name, date_col, start_year, start_month, end_year, end_month):
        _clear_one_period(self._cursor, table_name, date_col, start_year, start_month, end_year, end_month)

    def upsert(self, result, table_name, batch_size=5000):
        _batch_upsert_data(self._cursor, result, table_name, batch_size)

    def clear_months_after(self, table_name, date_col, start_year, start_month):
        _clear_months_after(self._cursor, table_name, date_col, start_year, start_month)


def _test():
    from global_vars import UatDbConfig
    db_config = UatDbConfig()
    writer = KuduResultWriter(db_config)
    table_name = "m111_level2_order_pred_result"
    writer.clear(table_name)


if __name__ == '__main__':
    _test()
