# _*_ coding: utf-8 _*_

"""
Customer list loader.

TODO: 第三期时，改成和型谱一样支持不同版本

Author: Genpeng Xu
"""

import os
import pandas as pd

# Own customized modules
from global_vars import CUSTOMER_LIST_DATA_DIR


class CustomerList:
    def __init__(self):
        self._customers = self._load_customer_list()
        self._white_customers = self._get_white_customers()

    def is_white_customer(self, customer_code):
        return customer_code in self._white_customers

    def _load_customer_list(self):
        filepath = os.path.join(CUSTOMER_LIST_DATA_DIR, "customer-list.txt")
        customers = pd.read_csv(filepath, sep='\t')
        customers['customer_code'] = customers.customer_code.astype(str).apply(lambda x: x.strip())
        return customers

    def _get_white_customers(self):
        white_customers = set(self._customers.customer_code)
        return white_customers

    @property
    def customers(self):
        return self._customers

    @property
    def white_customers(self):
        return self._white_customers


def test():
    customer_list = CustomerList()
    print(customer_list.is_white_customer('C0010840'))


if __name__ == '__main__':
    test()
