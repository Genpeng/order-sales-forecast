# _*_ coding: utf-8 _*_

"""

"""

import time
from impala.dbapi import connect


def create_cursor(config):
    conn = connect(host=config.host,
                   port=config.port,
                   database=config.database,
                   auth_mechanism=config.auth_mechanism,
                   user=config.user,
                   password=config.password)
    return conn.cursor()


def batch_upsert_data(cursor, df, table_name, batch_size=5000):
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
