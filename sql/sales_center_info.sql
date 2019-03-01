/* 销售中心维 */
SELECT 
	sales_cen_code
	,sales_cen_name
	,sales_cen_sort -- 销售中心类型
	,sales_region_code -- 大区编码
	,sales_region_name -- 大区名称
FROM DIM.DIM_SALES_CEN_D