SELECT DISTINCT item_code, sales_segment1_code, sales_segment1_name, sales_segment2_code, sales_segment2_name
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d

SELECT SUBSTR(CAST(period_wid AS VARCHAR), 1, 6) AS inv_date, sales_cen_code, customer_code, item_code, SUM(qty) AS inv_qty
FROM nsum.sum_dom_chnl_inv_d
WHERE period_wid >= 20170101 AND bd_code = 'M111' AND sales_segment1_code = 'DR'
GROUP BY inv_date, sales_cen_code, customer_code, item_code
ORDER BY inv_date ASC

--取每个月15号每个客户不同产品的库存
SELECT CAST(period_wid AS VARCHAR) AS inv_date, sales_cen_code, customer_code, item_code, SUM(qty) AS inv_qty
FROM nsum.sum_dom_chnl_inv_d
WHERE CAST(period_wid AS VARCHAR) LIKE '%15%' AND bd_code = 'M111' AND sales_segment1_code = 'DR'
GROUP BY inv_date, sales_cen_code, customer_code, item_code
ORDER BY inv_date ASC

SELECT
    period_wid
    ,sales_cen_code
    ,customer_code
    ,item_code
    ,SUM(quantity) AS qty
    ,SUM(cancel_qty) AS cancel_qty
FROM dwd.dwd_pl_lg_order_f
WHERE period_wid >= 20170101 AND org_code = 'M111' AND sales_segment1_code LIKE '%DR%'
GROUP BY period_wid, sales_cen_code, customer_code, item_code
ORDER BY period_wid ASC

-- 确认提货订单全流程采用哪个字段作为时间字段
SELECT 
	period_wid
	,cancel_date
	,checkup_date
	,to_checkup_date
	,order_date
	,created_on_dt
	,changed_on_dt
	,aux1_changed_on_dt
	,aux2_changed_on_dt
	,aux3_changed_on_dt
	,aux4_changed_on_dt
	,w_update_dt
	,w_insert_dt
	,center_checkup_date
	,estimate_consignment_stage
	,aps_promise_date
	,scheduled_completion_date
FROM dwd.dwd_pl_lg_order_f

SELECT 
	SUBSTR(order_date, 1, 10) AS order_date
	,sales_cen_code
	,customer_code
	,item_code
	,item_name
	,quantity
	,cancel_qty
FROM dwd.dwd_pl_lg_order_f
WHERE period_wid >= 20170101 AND org_code = 'M111' AND sales_segment1_code LIKE '%DR%'
ORDER BY order_date ASC

-- 分销取数
SELECT 
    SUBSTR(obnd_bill_date, 1, 10) AS bill_date
    ,sales_cen_code
    ,sell_customer_code
    ,item_code
    ,SUM(CASE WHEN confirm_qty <= 0 THEN 0 ELSE confirm_qty END) AS qty
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d
WHERE 
    SUBSTR(obnd_bill_date, 1, 10) >= '2017-01-01'
    AND SUBSTR(obnd_bill_date, 1, 10) <= '2018-11-30' 
    AND bd_code = 'M111' AND sales_segment1_code LIKE '%DR%'
GROUP BY bill_date, sales_cen_code, sell_customer_code, item_code
ORDER BY bill_date
