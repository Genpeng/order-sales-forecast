nohup python -u download_data.py > logs/update_history.log 2>&1 &
nohup python -u level1_order_predict_history.py -c configs/level1_order_predict_history_config.json > logs/update_history.log 2>&1 &
nohup python -u level2_order_predict_history.py -c configs/level2_order_predict_history_config.json > logs/update_history.log 2>&1 &
nohup python -u level3_order_predict_history.py -c configs/level3_order_predict_history_config.json > logs/update_history.log 2>&1 &
nohup python -u level3_dis_predict_history.py -c configs/level3_dis_predict_history_config.json > logs/update_history.log 2>&1 &
nohup python -u level3_inv_predict_history.py -c configs/level3_inv_predict_history_config.json > logs/update_history.log 2>&1 &