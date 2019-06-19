# _*_ coding: utf-8 _*_

"""


Author: Genpeng Xu
"""

from util.date_util import get_curr_date
from util.config_util import get_args, process_config


def main():
    # Step 0: Input variables
    # ============================================================================================ #

    pred_year, pred_month = 2019, 4


    # Step 1: Load configuration
    # ============================================================================================ #

    print("[INFO] Load configuration...")

    parser, config = None, None
    try:
        args, parser = get_args()  # get the path of configuration file
        config = process_config(args.config)
    except Exception as e:
        print(e)
        if parser:
            parser.print_help()
        exit(0)

    print("[INFO] Parsing finished!")
    print("[INFO] Months need to be predicted are:", config.pred_months)

    # Step 2: Load data
    # ============================================================================================ #

    curr_year, curr_month, _ = get_curr_date()
    data_file_flag = "%d-%02d" % (curr_year, curr_month)
    inv_path = "/data/aidev/order-sales-forecast/data/inv/%s/m111-sku-inv_%s.txt" % tuple([data_file_flag] * 2)
    column_names = ['order_date',
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
                    'item_price']
    inv = pd.read_csv(inv_path, sep='\u001e', header=None, names=column_names, parse_dates=[0])
    inv.drop(columns=['bu_code',
                      'bu_name',
                      'region_code',
                      'region_name'], inplace=True)
    inv = inv.loc[inv.first_cate_code.isin(all_cates)]
    inv = inv.sort_values(by='order_date').reset_index(drop=True)
    inv['inv_qty'] = inv.inv_qty / 10000
    inv['inv_amount'] = inv.inv_amount / 10000

    # Step
    # ============================================================================================ #

    # Step
    # ============================================================================================ #

    # Step
    # ============================================================================================ #


if __name__ == '__main__':
    main()
