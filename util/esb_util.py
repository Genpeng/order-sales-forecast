# _*_ coding: utf-8 _*_

"""
Some utility functions about pushing data to ESB.

Author: Genpeng Xu
"""

import time
import json
import requests

# Own customized modules
from util.date_util import get_curr_date


def generate_serial_number():
    dt_str = "%d%02d%02d" % get_curr_date()
    return "m111-order-%s%s" % (dt_str, str(time.time())[-4:])


def push_data(df, serial_no, url):
    df_json_str = df.to_json(orient='records', force_ascii=False)
    df_json_str = "{\"hData\":%s}" % df_json_str
    json_obj = {
        "Envelope": {
            "Header": {
                "requestHeader": {
                    "version": "1.0",
                    "serialNo": serial_no,
                    "requestId": "BDOF",
                    "namespace": "http://www.midea.com/afp/AfpMassDataImportService/v1"
                }
            },
            "Body": {
                "DataImport": {
                    "serialNo": serial_no,
                    "iFaceCode": "GAPS-BDOF-001",
                    "source_code": "BDOF",
                    "data": df_json_str
                }
            }
        }
    }
    json_str = json.dumps(json_obj)
    response = requests.post(url, data=json_str)
    response_obj = json.loads(response.text)
    if response_obj["Envelope"]["Body"]["DataImportResponse"]["isSuccess"]:
        return response_obj, True
    else:
        return response_obj, False


def push_to_esb(df, esb_url):
    print("[INFO] Start push to ESB...")
    serial_no = generate_serial_number()
    try_time, success_mark = 0, False
    while not success_mark:
        time.sleep(5)
        _, success_mark = push_data(df, serial_no, esb_url)
        try_time += 1
        if try_time > 10:
            raise Exception("[INFO] Fail to push data to ESB!!!")
    print("[INFO] Push finished! ( ^ _ ^ ) V")
