# _*_ coding: utf-8 _*_

import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ORDER_DATA_DIR = os.path.join(DATA_DIR, 'order')
DIS_DATA_DIR = os.path.join(DATA_DIR, 'dis')
INV_DATA_DIR = os.path.join(DATA_DIR, 'inv')

# 考虑的品类有：烟机、灶具、消毒柜、洗碗机、电热、燃热、饮水机、净水机
ALL_CATE_CODES = ['CRYJ', 'CRZJ', 'CRXDG', 'CRXWJ', 'DR', 'RR', 'YSJ', 'JSJ']
CHU_DIAN_CATE_CODES = ['CRYJ', 'CRZJ', 'CRXDG', 'CRXWJ']  # 厨电事业部包含的品类
RE_SHUI_QI_CATE_CODES = ['DR', 'RR', 'YSJ', 'JSJ']  # 热水器事业部包含的品类

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
DIS_DATA_COLUMNS_NAMES = [
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


if __name__ == '__main__':
    print(ROOT_DIR)
    print(DATA_DIR)
    print(ORDER_DATA_DIR)
    print(os.listdir(ORDER_DATA_DIR))
