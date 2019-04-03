# _*_ coding: utf-8 _*_

"""
Some utility functions about date.

Author: Genpeng Xu
"""


def get_curr_date():
    import datetime
    curr_date = datetime.datetime.now()
    return curr_date.year, curr_date.month, curr_date.day


def get_days_of_month(year, month):
    import calendar
    return calendar.monthrange(year, month)[1]


def get_month(year, month, offset=-1):
    if month + offset <= 0:
        month += 12 + offset
        year -= 1
    else:
        month += offset
    return year, month


def get_pre_months(year, month, left_bound='2015-09'):
    start_year, start_month = int(left_bound.split('-')[0]), int(left_bound.split('-')[1])
    all_months = []
    if year == start_year and month >= start_month:
        months = ['%d-%02d' % (start_year, m) for m in range(start_month, month + 1)]
        all_months.extend(months)
    elif year == start_year + 1:
        start_year_months = ['%d-%02d' % (start_year, m) for m in range(start_month, 13)]
        all_months.extend(start_year_months)
        curr_year_months = ['%d-%02d' % (year, m) for m in range(1, month + 1)]
        all_months.extend(curr_year_months)
    elif year > start_year + 1:
        start_year_months = ['%d-%02d' % (start_year, m) for m in range(start_month, 13)]
        all_months.extend(start_year_months)
        for y in range(start_year + 1, year):
            months = ['%d-%02d' % (y, m) for m in range(1, 13)]
            all_months.extend(months)
        curr_year_months = ['%d-%02d' % (year, m) for m in range(1, month + 1)]
        all_months.extend(curr_year_months)
    else:
        raise Exception("[ERROR] The input date is earlier than the start date!!!")
    return all_months


def main():
    # 测试 get_days_of_month
    # print(get_days_of_month(2019, 3))

    # 测试 get_pre_months
    # print(get_pre_months(2019, 4))

    # 测试 get_month
    # year, month = 2019, 4
    # print(get_month(year, month, offset=-4))
    # print(year, month)

    # 测试 get_curr_date
    print(get_curr_date())


if __name__ == '__main__':
    main()
