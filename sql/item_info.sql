/* 物料头维 */
SELECT 
	item_code
	,item_name
	,brand_code
	,brand_code
	,start_date_active
	,end_date_active
	,abilitylevel_code
	,abilitylevel
	,direct_code
	,direct_name
	,category_code
	,category_name
	,segment1_code
	,segment1_name
	,segment2_code
	,segment2_name
	,segment3_code
	,segment3_name
	,life_cycle_code
	,life_cycle_name
FROM DIM.DIM_ITEM_HEAD_D

/* 产品生命周期 */
SELECT *
FROM DIM.DIM_PRODDUCT_MANAGER_D -- 没有这张表