#!/bin/bash
source activate xgp_py3
python download_data.py
python level1_order_predict_future.py -c configs/level1_order_predict_future_config.json
python level2_order_predict_future.py -c configs/level2_order_predict_future_config.json
python level3_order_predict_future.py -c configs/level3_order_predict_future_config.json
python level3_dis_predict_future.py -c configs/level3_dis_predict_future_config.json
python level3_inv_predict_future.py -c configs/level3_inv_predict_future_config.json