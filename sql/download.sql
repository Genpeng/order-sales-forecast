SELECT COUNT(*) FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'

-- 下载数据
SELECT  creation_date, manage_org_wid, sales_region_wid, region_wid, 
        sales_cen_wid, customer_wid, sales_category_wid, item_wid, 
        item_code, item_name, item_price, item_uom, 
        item_qty, productform, energyeffratiing, brand, 
        frequencytype, sales_segment1_code, sales_segment1_name, sales_segment2_code, 
        return_qty, return_received_qty, return_flag
FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'