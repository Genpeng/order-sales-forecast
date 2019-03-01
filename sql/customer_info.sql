SELECT DISTINCT 
	CUST_CODE, 
	CUST_NAME, 
	TREE_LEVEL, 
	SSETS_OF_BOOKS_ID, 
	BSETS_OF_BOOKS_ID, 
	CAT_ID, 
	ORG_ID, 
	ORG_CODE, 
	ORG_NAME, 
	USABLE 
-- FROM GCCS.CCS_SETS_OF_BOOKS_RELAT
FROM gccs.ccs_sets_of_books_relat
WHERE MAST_CUST = 2 and tree_level = 1 
--AND USABLE = 2

/* 代理商信息 */
SELECT DISTINCT 
	cust_code -- 代理商编码（上游账套客户编码）
	,cust_name -- 代理商名称（上游账套客户名称）
	,tree_level -- 代理商层级
	,ssets_of_books_id -- 代理商ID（上游账套ID）
	,bsets_of_books_id -- 分销商ID（下游账套ID）
	,cat_id -- 品类ID
	,org_id
	,org_code
	,org_name
	,usable -- 是否有效
FROM gccs.ccs_sets_of_books_relat