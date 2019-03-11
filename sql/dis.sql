/* 
# =========== 关于分销数据取数的说明 =========== #
1. 需求简述
由于对数据建模需要较多数据，所以时间跨度最好尽可能的大，最好是从【2015年1月至
2019年2月】的数据。另外，现在项目的需求是对代理商的SKU进行预测，因此数据的粒度
必须到达【代理商（层级为一级）】，即每一条记录必须有事业部编码、销售中心编码、
代理商编码、产品编码。

dis_date	string	分销日期
bd_code	string	事业部编码
bd_name	string	事业部名称
sales_cen_code	string	销售中心编码
sales_cen_name	string	销售中心名称
sell_customer_code	string	卖方客户编码
sell_customer_name	string	卖方客户名称
item_code	string	物料编码
item_name	string	物料名称
bill_qty	decimal(30,8)	开单数量
confirm_qty	decimal(30,8)	出库确认数量
standard_price	decimal(30,8)	标准单价
standard_amount	decimal(30,8)	销售金额

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
*/