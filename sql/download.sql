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

