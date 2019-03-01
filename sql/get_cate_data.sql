/* ================ 取总部品类提货数据（月份） ================ */
SELECT 
    SUBSTR(order_date, 1, 7) AS order_date
	,sales_segment1_code -- 一级营销大类
    ,(SUM(quantity) - SUM(cancel_qty)) AS qty
FROM dwd.dwd_pl_lg_order_f
WHERE org_code = 'M111'
GROUP BY order_date, sales_segment1_code
ORDER BY order_date ASC

/* ================ 取品类的名称 ================ */
SELECT DISTINCT 
	sales_segment1_code
	,sales_segment1_name
FROM dwd.dwd_pl_lg_order_f
WHERE org_code = 'M111'

/* ================ 取总部品类的库存数据 ================ */
SELECT 
    period_wid
    ,sales_segment1_code -- 一级营销大类
    ,SUM(qty) AS qty
FROM nsum.sum_dom_chnl_inv_d
WHERE bd_code = 'M111'
GROUP BY period_wid, sales_segment1_code
ORDER BY period_wid ASC

/* ================ 取总部品类的分销数据 ================ */
SELECT 
    SUBSTR(obnd_bill_date, 1, 10) AS dis_date
    ,sales_segment1_code
    ,SUM(confirm_qty) AS qty
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d
WHERE bd_code = 'M111'
GROUP BY dis_date, sales_segment1_code
ORDER BY dis_date ASC