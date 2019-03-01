分销： select * from DW.dw_so_sales_f F WHERE F.source_code = 'CCS' --- 含17年及之前数据 
分销： DM.dm_dom_chnl_inv_out_bill_anal_d --- 只含18年之后数据 
库存： select * from DW.dw_inv_onhand_quantity_f F WHERE F.source_code = 'CCS' --- 含17年及之前数据 
库存： DM.dm_dom_chnl_inv_anal_d --- 只含18年之后数据