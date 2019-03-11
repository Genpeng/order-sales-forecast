/* ============= 历史分销数据 ============= */
SELECT * FROM dw.dw_so_sales_f
WHERE source_code = 'CCS' -- 含17年及之前数据

SELECT 
	period_wid -- bigint
	,order_date -- string
	,sale_center_wid
	,sales_center
	,customer_wid
	,crm_customer_code
	,old_customer_code
	,item_wid
	,item_code
	,category2_wid
	,region_wid -- 不知道指销售中心还是代理商的地区 为空
	,unit_price
	,quantity
FROM dw.dw_so_sales_f
WHERE source_code = 'CCS'
ORDER BY period_wid ASC

SELECT 
	order_date -- string
	,sales_center -- 销售中心编码
	,crm_customer_code -- 代理商编码
	,item_code -- 产品编码
	,unit_price -- 单价
	,quantity -- 数量
	,amount -- 销售金额
FROM dw.dw_so_sales_f
WHERE source_code = 'CCS' AND manage_org_wid = 361 AND direct_wid = 1
ORDER BY order_date ASC

SELECT 
	order_date
	,item_code
	,SUM(quantity) AS qty
	

/* ============= 历史库存数据 ============= */
SELECT * FROM dw.dw_inv_onhand_quantity_f
WHERE source_code = 'CCS'

SELECT 
	period_wid
	,sales_cen_code
	,sales_cen_name
	,customer_wid -- 代理商维ID
	,organization_wid -- 库存组织维ID
	,organization_id -- 库存组织ID
	,item_wid
	,item_code
	,direct_wid -- 内外销
	,category_wid
	,sales_region_code
	,sales_region_name
	,onhand_type -- 库存类型
	,quantity
FROM dw.dw_inv_onhand_quantity_f
WHERE source_code = 'CCS'
ORDER BY period_wid ASC
