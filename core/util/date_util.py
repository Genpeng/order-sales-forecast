# _*_ coding: utf-8 _*_


def get_curr_date():
    import datetime
    curr_date = datetime.datetime.now()
    return curr_date.year, curr_date.month, curr_date.day


def infer_month(year, month, offset=-1):
    if month + offset <= 0:
        month = month + 12 + offset
        year -= 1
    elif month + offset >= 13:
        month = month + offset - 12
        year += 1
    else:
        month += offset
    return year, month