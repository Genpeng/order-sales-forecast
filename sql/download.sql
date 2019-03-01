SELECT COUNT(*) FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'

SELECT COUNT(*) FROM dw.dwd_pl_so_order_f
WHERE creation_date >= UNIX_TIMESTAMP('2018-01-01')

SELECT DISTINCT item_wid
FROM dw.dwd_pl_so_order_f

-- 下载数据
SELECT  creation_date, manage_org_wid, sales_region_wid, region_wid, 
        sales_cen_wid, customer_wid, sales_category_wid, item_wid, 
        item_code, item_name, item_price, item_uom, 
        item_qty, productform, energyeffratiing, brand, 
        frequencytype, sales_segment1_code, sales_segment1_name, sales_segment2_code, 
        return_qty, return_received_qty, return_flag
FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'
ORDER BY creation_date ASC

-- item_info
SELECT DISTINCT item_wid, item_code, item_name, sales_category_wid, 
                item_price, item_uom, productform, energyeffratiing, 
                brand, frequencytype, sales_segment1_code, sales_segment1_name
                sales_segment2_code
FROM dw.dwd_pl_so_order_f

-- item_info
SELECT DISTINCT item_code, sales_category_wid, productform, energyeffratiing, frequencytype
FROM dw.dwd_pl_so_order_f

-- center_info
SELECT DISTINCT sales_cen_wid, region_wid, sales_region_wid
FROM dw.dwd_pl_so_order_f

-- sales
SELECT  creation_date, sales_cen_wid, customer_wid, item_code, 
        item_qty, return_qty
FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'
ORDER BY creation_date ASC

-- sales
SELECT  SUBSTR(creation_date, 1, 10) AS creation_date, sales_cen_wid, customer_wid, item_code, 
        SUM(item_qty) AS item_qty, SUM(CASE WHEN return_qty IS NULL THEN 0 ELSE return_qty END) as return_qty
FROM dw.dwd_pl_so_order_f
WHERE creation_date >= '2018-01-01'
GROUP BY creation_date, sales_cen_wid, customer_wid, item_code
ORDER BY creation_date ASC

-- 产品品类
SELECT ITEM_CODE,ORGANIZATION_ID BD_CODE,CATEGORY_WID,SEGMENT1_CODE,SEGMENT1_NAME,SEGMENT2_CODE,SEGMENT2_NAME ,
       ROW_NUMBER() OVER(PARTITION BY ITEM_CODE,ORGANIZATION_ID ORDER BY CASE WHEN SEGMENT2_NAME IS NOT NULL THEN 1 ELSE 2 END ASC ) AS RN
FROM DW.DW_PUB_ITEM_CATEGORY_RELA DPI
WHERE DPI.CATEGORY_SET_ID = 3  -- 指的是营销分类

SELECT 	item_code, organization_id AS bd_code, category_wid, 
		segment1_code, segment1_name, segment2_code, segment2_name, 
		ROW_NUMBER() OVER(PARTITION BY item_code, organization_id ORDER BY CASE WHEN segment2_name IS NOT NULL THEN 1 ELSE 2 END ASC) AS rn
FROM dw.dw_pub_item_category_rela AS dpi
WHERE dpi.category_set_id = 3

-- 零售数据（门店 -> 用户）
SELECT 	period_id, bd_code, bd_name, sales_region_code, sales_region_name, 
		region_code, region_name, terminal_code, terminal_name, 
		terminal_type_code, terminal_type_name, sales_segment1_code, sales_segment1_name, 
		sales_segment2_code, sales_segment2_name, item_code, item_name, 
		freq_type_code, freq_type_name, customer_code, customer_name, 
		customer_code2, customer_name2, sales_type, prodform, prodpos, 
		brand_name, quantity, ou_name, top1500
FROM dm.dm_sto_sales_dtl
WHERE source_code = 'MMP'

-- 查询
SELECT * FROM DW.DWD_PL_SO_ORDER_F
WHERE belong_to_customer_code = 'C0013060'

-- 取分销数据（一级客户 -> 经销商）
SELECT 	bd_code, bd_name, sales_cen_code, sales_cen_name, 
		region_code, region_name, region_level2_name, region_level3_name, 
		region_level4_name, item_code, item_name, is_wl, 
		prod_type_code, prod_type_name, sales_segment1_code, sales_segment1_name, 
		sales_segment2_code, sales_segment2_name, sell_customer_code, sell_customer_name, 
		confirm_qty, submit_date, w_insert_dt, period_wid, obnd_bill_date, check_date
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d

SELECT period_wid, submit_date, w_insert_dt, obnd_bill_date, check_date
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d

SELECT DISTINCT bd_code, bd_name, sales_cen_code, sales_cen_name, 
				sell_customer_code, sell_customer_name
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d
WHERE sell_customer_code = 'C0004719'

SELECT 	SUBSTR(obnd_bill_date, 1, 10) AS obnd_bill_date, bd_code, sales_cen_code, sell_customer_code, item_code, 
		SUM(confirm_qty) AS qty
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d
WHERE obnd_bill_date >= '2017-01-01'
GROUP BY obnd_bill_date, bd_code, sales_cen_code, sell_customer_code, item_code
ORDER BY obnd_bill_date ASC

SELECT DISTINCT bd_code, item_code
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d

-- 渠道库存
SELECT 	w_insert_dt, period_wid, 
        bd_code, bd_name, sales_cen_code, sales_cen_name, 
		customer_code, customer_name, item_code, item_name, 
		sales_segment1_code, sales_segment1_name, sales_segment2_code, sales_segment2_name, 
		inv_cdde, inv_name, inv_type, is_logis, qty
FROM dm.dm_dom_chnl_inv_anal_d

SELECT  period_wid, bd_code, bd_name, sales_cen_code, sales_cen_name, 
        customer_code, customer_name, customer_type, usable, is_cancel_customer, chnl_lvl, 
        item_code, item_name, is_wl, 
        inv_cdde, inv_name, inv_type, is_logis, qty
FROM dm.dm_dom_chnl_inv_anal_d

SELECT  period_wid, bd_code, bd_name, sales_cen_code, sales_cen_name, 
        customer_code, customer_name, customer_type, usable, is_cancel_customer, chnl_lvl, 
        item_code, item_name, is_wl, 
        inv_cdde, inv_name, inv_type, is_logis, qty
FROM dm.dm_dom_chnl_inv_anal_d
WHERE period_wid >= 20180101
ORDER BY period_wid



