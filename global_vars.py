# _*_ coding: utf-8 _*_

"""
All the global variables needed for this project.

Author: Genpeng Xu
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

HDFS_ROOT_DIR = "/user/hive/warehouse/ai_orders_predict.db/"

ORDER_TABLE_NAME = "init_pl_order"
DIS_TABLE_NAME = "init_channel_sales"
INV_TABLE_NAME = "init_channel_inv"
PLAN_TABLE_NAME = "dm_order_predict_pln_m"

ORDER_DATA_DIR = os.path.join(ROOT_DIR, "data/order")
DIS_DATA_DIR = os.path.join(ROOT_DIR, "data/dis")
INV_DATA_DIR = os.path.join(ROOT_DIR, "data/inv")
PLAN_DATA_DIR = os.path.join(ROOT_DIR, "data/plan")
CUSTOMER_LIST_DATA_DIR = os.path.join(ROOT_DIR, "data")

# data columns names
ORDER_DATA_COLUMN_NAMES = [
    'order_date',
    'org_code',
    'sales_cen_code',
    'sales_cen_name',
    'sales_region_code',
    'sales_region_name',
    'customer_code',
    'customer_name',
    'region_code',
    'region_name',
    'province',
    'city',
    'district',
    'road',
    'item_code',
    'item_name',
    'first_cate_code',
    'first_cate_name',
    'second_cate_code',
    'second_cate_name',
    'fin_segment1_code',
    'fin_segment1_name',
    'fin_segment2_code',
    'fin_segment2_name',
    'channel_name',
    'sales_chan_name',
    'project_flag',
    'item_price',
    'ord_qty',
    'ord_amount',
    'received_qty',
    'return_qty'
]
DIS_DATA_COLUMN_NAMES = [
    'order_date',
    'bu_code',
    'bu_name',
    'customer_code',
    'customer_name',
    'sales_cen_code',
    'sales_cen_name',
    'sales_region_code',
    'sales_region_name',
    'region_code',
    'region_name',
    'province',
    'city',
    'district',
    'road',
    'item_code',
    'item_name',
    'first_cate_code',
    'first_cate_name',
    'second_cate_code',
    'second_cate_name',
    'customer_type',
    'is_usable',
    'channel_name',
    'channel_level',
    'sales_chan_name',
    'dis_qty',
    'item_price',
    'dis_amount'
]
INV_DATA_COLUMN_NAMES = [
    'order_date',
    'bu_code',
    'bu_name',
    'sales_cen_code',
    'sales_cen_name',
    'sales_region_code',
    'sales_region_name',
    'region_code',
    'region_name',
    'province',
    'city',
    'district',
    'customer_code',
    'customer_name',
    'customer_type',
    'is_usable',
    'channel_name',
    'channel_level',
    'sales_chan_name',
    'item_code',
    'item_name',
    'first_cate_code',
    'first_cate_name',
    'second_cate_code',
    'second_cate_name',
    'inv_qty',
    'inv_amount',
    'item_price'
]
PLAN_DATA_COLUMN_NAMES = [
    'bd_code',
    'bd_name',
    'order_type_name',
    'period_code',
    'begin_date',
    'end_date',
    'customer_code',
    'customer_name',
    'sales_center_code',
    'sales_center_name',
    'item_code',
    'item_name',
    'plan_mode',
    'month_plan_qty',
    'm2_month_plan_qty',
    'm3_month_plan_qty',
    'w_insert_dt'
]
ITEM_LIST_COLUMN_NAMES = [
    'id',
    'item_code',
    'item_name',
    'first_cate_code',
    'state',
    'manu_code',
    'need_calc',
    'is_highend',
    'appear_date',
    'delisting_date',
    'scheduled_delisting_date',
    'six_eighteen_main_product_date',
    'double_eleven_main_product_date',
    'last_update_date',
    'last_updated_by',
    'creation_date',
    'created_by',
    'version'
]

SUPPORTED_CATE_CODES = {
    "CRZJ",
    "CRXDG",
    "CRYJ",
    "JSJ",
    "YSJ",
    "RR",
    "DR",
    "CRXWJ"
}
SUPPORTED_CATE_NAMES = {
    '净水机',
    '饮水机',
    '烟机',
    '灶具',
    '洗碗机',
    '消毒柜',
    '电热水器',
    '燃气热水器'
}
CATE_CODE_2_CATE_NAME = {
    "CRZJ": '灶具',
    "CRXDG": '消毒柜',
    "CRYJ": '烟机',
    "JSJ": '净水机',
    "YSJ": '饮水机',
    "RR": '燃气热水器',
    "DR": '电热水器',
    "CRXWJ": '洗碗机'
}
CATE_NAME_2_CATE_CODE = {
    '灶具': "CRZJ",
    '消毒柜': "CRXDG",
    '烟机': "CRYJ",
    '净水机': "JSJ",
    '饮水机': "YSJ",
    '燃气热水器': "RR",
    '电热水器': "DR",
    '洗碗机': "CRXWJ"
}


class SitDbConfig:
    def __init__(self):
        self.host = '10.18.25.92'
        self.port = 31051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


class UatDbConfig:
    def __init__(self):
        self.host = '10.18.25.92'
        self.port = 21051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


class ProdDbConfig:
    def __init__(self):
        self.host = '10.18.25.72'
        self.port = 21051
        self.database = 'ai_orders_predict'
        self.auth_mechanism = 'PLAIN'
        self.user = 'aiop_bind'
        self.password = '1Ye8Xl1Td0'


SIT_DB_CONFIG = {
    "host": "10.18.25.92",
    "port": 31051,
    "database": "ai_orders_predict",
    "auth_mechanism": "PLAIN",
    "user": "aiop_bind",
    "password": "1Ye8Xl1Td0"
}
UAT_DB_CONFIG = {
    "host": "10.18.25.92",
    "port": 21051,
    "database": "ai_orders_predict",
    "auth_mechanism": "PLAIN",
    "user": "aiop_bind",
    "password": "1Ye8Xl1Td0"
}
PROD_DB_CONFIG = {
    "host": "10.18.25.72",
    "port": 21051,
    "database": "ai_orders_predict",
    "auth_mechanism": "PLAIN",
    "user": "aiop_bind",
    "password": "1Ye8Xl1Td0"
}

ITEM_LIST_MYSQL_CONFIG = {
    "host": "3306-W-BD-UAT-TLY01-MYC5.service.dcnh.consul",
    "port": 3306,
    "database": "apd",
    "user": "readonly",
    "mode": "readonly"
}

if __name__ == '__main__':
    print(ROOT_DIR)
    print(ORDER_DATA_DIR)
    print(DIS_DATA_DIR)
    print(INV_DATA_DIR)
