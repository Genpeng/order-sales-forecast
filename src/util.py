# _*_ coding: utf-8 _*_
"""
Some auxiliary classes or functions.

Author: StrongXGP (xgp1227@gmail.com)
Date:	2018/11/16
"""


class Inspector:
    @staticmethod
    def find_outliers(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()[1:]
        d, outliers = {}, []
        for line in lines:
            cols = line.split(',')
            k, v = cols[0], cols[1]
            if k in d:
                if v != d[k]:
                    outliers.append(k)
            else:
                d[k] = v
        return outliers


def main():
    # inspector = Inspector()
    print(Inspector.find_outliers('../data/salescen-region.csv'))


if __name__ == '__main__':
    main()
