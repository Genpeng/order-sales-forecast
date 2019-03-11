/* 
### 订单需要的字段 ###
order_date	string	订单日期
org_code	string	组织代码（事业部代码）
org_name	string	组织名称（事业部名称）
sales_cen_code	string	营销中心编码
sales_cen_name	string	营销中心名称
customer_code	string	客户编码
customer_name	string	客户名称
sales_region_code	string	管理区域代码
sales_region_name	string	管理区域名称
region_code	string	地理区域编码
region_name	string	地理区域名称
item_code	string	产品编码
item_name	string	产品名称
sales_segment1_code	string	营销大类代码
sales_segment1_name	string	营销大类名称
sales_segment2_code	string	营销小类代码
sales_segment2_name	string	营销小类名称
fin_segment1_code	string	财务大类代码
fin_segment1_name	string	财务大类名称
fin_segment2_code	string	财务小类代码
fin_segment2_name	string	财务小类名称
quantity	decimal(30,8)	申请数量
cancel_qty	decimal(30,8)	取消数量（物流回写）
list_price	decimal(30,8)	单价
*/

-- SQL语句
SELECT 
    order_date
    ,org_code
    ,org_name
    ,sales_cen_code
    ,sales_cen_name
    ,customer_code
    ,customer_name
    ,sales_region_code
    ,sales_region_name
    ,region_code
    ,region_name
    ,item_code
    ,item_name
    ,sales_segment1_code
    ,sales_segment1_name
    ,sales_segment2_code
    ,sales_segment2_name
    ,fin_segment1_code
    ,fin_segment1_name
    ,fin_segment2_code
    ,fin_segment2_name
    ,quantity
    ,cancel_qty
    ,list_price
FROM dwd.dwd_pl_lg_order_f