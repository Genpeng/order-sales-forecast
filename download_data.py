# _*_ coding: utf-8 _*_

"""
Download order, distribution or inventory data.

Author: Genpeng Xu
"""

import os
import shutil
from datetime import datetime
from hdfs.client import HdfsError
from hdfs.client import InsecureClient

# Own customized modules
from util.date_util import get_curr_date, infer_month
from global_vars import (HDFS_ROOT_DIR, ORDER_DATA_DIR, DIS_DATA_DIR,
                         INV_DATA_DIR, PLAN_DATA_DIR, ORDER_TABLE_NAME,
                         DIS_TABLE_NAME, INV_TABLE_NAME, PLAN_TABLE_NAME)


def delete_all(delete_dir):
    """Recursively delete all files and directories under the specified directory."""
    all_files = os.listdir(delete_dir)
    for f in all_files:
        filepath = os.path.join(delete_dir, f)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as e:
            print(e)
            return False
    return True


def download(client, table_name, year, month):
    partition_flag = "part_dt=%d-%02d" % (year, month)
    file_dir = os.path.join(HDFS_ROOT_DIR, table_name, partition_flag)
    hdfs_filenames = client.list(file_dir)
    hdfs_filepaths = [os.path.join(file_dir, filename) for filename in hdfs_filenames]

    save_flag = "%d-%02d" % (year, month)
    if table_name == ORDER_TABLE_NAME:
        save_dir = os.path.join(ORDER_DATA_DIR, save_flag)
        final_filename = "m111-%s_%d-%02d.txt" % ('order', year, month)
    elif table_name == DIS_TABLE_NAME:
        save_dir = os.path.join(DIS_DATA_DIR, save_flag)
        final_filename = "m111-%s_%d-%02d.txt" % ('dis', year, month)
    elif table_name == INV_TABLE_NAME:
        save_dir = os.path.join(INV_DATA_DIR, save_flag)
        final_filename = "m111-%s_%d-%02d.txt" % ('inv', year, month)
    elif table_name == PLAN_TABLE_NAME:
        save_dir = os.path.join(PLAN_DATA_DIR, save_flag)
        final_filename = "m111-%s_%d-%02d.txt" % ('plan', year, month)
    else:
        raise Exception("[ERROR] The downloaded data type is illegal! Please check the `data_flag.")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        delete_all(save_dir)

    for i, f in enumerate(hdfs_filepaths):
        save_filepath = os.path.join(save_dir, "%d.txt" % (i + 1))
        try:
            client.download(f, save_filepath)
        except HdfsError:
            print("[ERROR] %s download failed! Try again!" % f)
            os.remove(save_filepath)
            try:
                client.download(f, save_filepath)
            except HdfsError:
                print("[ERROR] Cannot download %s successfully!")
                return

    final_filepath = os.path.join(save_dir, final_filename)
    save_filepaths = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
    with open(final_filepath, 'w', encoding='utf-8') as fout:
        for filepath in save_filepaths:
            with open(filepath, 'r', encoding='utf-8') as fin:
                fout.writelines(fin.readlines())
            os.remove(filepath)


if __name__ == '__main__':
    curr_year, curr_month, _ = get_curr_date()

    # uat_url = "http://10.18.104.201:50070"
    # uat_client = InsecureClient(uat_url, user='caojun1')
    # download(uat_client, ORDER_TABLE_NAME, curr_year, curr_month)
    # download(uat_client, DIS_TABLE_NAME, curr_year, curr_month)
    # download(uat_client, INV_TABLE_NAME, curr_year, curr_month)
    # if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
    #     p_y, p_m = infer_month(curr_year, curr_month, -1)
    #     download(uat_client, PLAN_TABLE_NAME, curr_year, curr_month)
    # else:
    #     download(uat_client, PLAN_TABLE_NAME, curr_year, curr_month)

    prod_url = "http://10.18.104.29:50071"
    prod_client = InsecureClient(prod_url, user='caojun1')
    download(prod_client, ORDER_TABLE_NAME, curr_year, curr_month)
    download(prod_client, DIS_TABLE_NAME, curr_year, curr_month)
    download(prod_client, INV_TABLE_NAME, curr_year, curr_month)
    if datetime.now() < datetime(curr_year, curr_month, 16, 13, 0, 0):
        p_y, p_m = infer_month(curr_year, curr_month, -1)
        download(prod_client, PLAN_TABLE_NAME, p_y, p_m)
    else:
        download(prod_client, PLAN_TABLE_NAME, curr_year, curr_month)
