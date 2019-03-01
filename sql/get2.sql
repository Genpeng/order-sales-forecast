SELECT
    part_dt
    ,bd_code
    ,product_code
    ,sales_segment1_code
    ,inv_on_hand_qty
FROM dm.dm_pl_cp_inventory_dtl
ORDER BY part_dt

SELECT
    SUBSTR(part_dt, 1, 7) AS inv_date
    ,bd_code
    ,sales_segment1_code
    ,SUM(inv_on_hand_qty) AS inv_qty
FROM dm.dm_pl_cp_inventory_dtl
GROUP BY inv_date, bd_code, sales_segment1_code
ORDER BY inv_date

