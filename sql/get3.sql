-- 目标数据
SELECT
    period_wid
    ,bd_code
    ,bd_name
    ,sales_cen_code
    ,sales_cen_name
    ,customer_code
    ,customer_name
    ,item_code
    ,picking_set_mon
    ,picking_amount_mon
    ,distribution_set_mon
    ,distribution_amount_mon
    ,picking_set_year
    ,picking_amount_year
    ,distribution_set_year
    ,distribution_amount_year
FROM dw.dw_dom_pln_sales_pace_d
WHERE bd_code = 'M107'
ORDER BY period_wid DESC

-- 总部SKU粒度取数
SELECT 
    SUBSTR(order_date, 1, 10) AS order_date
    ,item_code
    ,SUM(quantity) AS qty
    ,SUM(cancel_qty) AS cancel_qty
FROM dwd.dwd_pl_lg_order_f
GROUP BY order_date, item_code
ORDER BY order_date DESC

-- 取厨热的提货订单数据
SELECT 
    SUBSTR(order_date, 1, 10) AS order_date
    ,sales_segment1_code -- 一级营销大类
    ,item_code -- 产品编码
    ,(SUM(quantity) - SUM(cancel_qty)) AS qty
FROM dwd.dwd_pl_lg_order_f
WHERE org_code = 'M111'
GROUP BY order_date, sales_segment1_code, item_code
ORDER BY order_date DESC

-- 取厨热的分销数据
SELECT 
    SUBSTR(obnd_bill_date, 1, 10) AS dis_date
    ,item_code
    ,SUM(confirm_qty) AS qty
FROM dm.dm_dom_chnl_inv_out_bill_dtl_d
WHERE bd_code = 'M111'
GROUP BY dis_date, item_code
ORDER BY dis_date ASC

-- 取厨热的库存数据（代理商）
SELECT 
    period_wid
    ,item_code
    ,SUM(qty) AS qty
FROM nsum.sum_dom_chnl_inv_d
WHERE bd_code = 'M111'
GROUP BY period_wid, item_code
ORDER BY period_wid ASC

-- 取厨热的总部库存数据
SELECT 
    period_wid
    ,product_code
    ,SUM(inv_usable_qty) AS qty
FROM dm.dm_pl_cp_inventory_dtl
WHERE bd_code = 'M111'
GROUP BY period_wid, product_code
ORDER BY period_wid ASC