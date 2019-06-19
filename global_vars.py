# _*_ coding: utf-8 _*_
"""
All the global variables needed for this project.

Author: Genpeng Xu
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER_DATA_DIR = os.path.join(ROOT_DIR, "data/order")
DIS_DATA_DIR = os.path.join(ROOT_DIR, "data/dis")
INV_DATA_DIR = os.path.join(ROOT_DIR, "data/inv")

# data columns names
ORDER_DATA_COLUMN_NAMES = []
DIS_DATA_COLUMNS_NAMES = []
INV_DATA_COLUMN_NAMES = [
    'order_date', 'bu_code', 'bu_name', 'sales_cen_code', 'sales_cen_name',
    'sales_region_code', 'sales_region_name', 'region_code', 'region_name',
    'province', 'city', 'district', 'customer_code', 'customer_name',
    'customer_type', 'is_usable', 'channel_name', 'channel_level',
    'sales_chan_name', 'item_code', 'item_name', 'first_cate_code',
    'first_cate_name', 'second_cate_code', 'second_cate_name', 'inv_qty',
    'inv_amount', 'item_price'
]


def main():
    print(ROOT_DIR)
    print(ORDER_DATA_DIR)
    print(DIS_DATA_DIR)
    print(INV_DATA_DIR)


if __name__ == '__main__':
    main()
