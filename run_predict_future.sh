nohup python -u download_data.py > logs/download_data.log 2>&1 &
nohup python -u level1_order_predict_future.py -c configs/level1_order_predict_future_config.json > logs/level1_order_predict_future.log 2>&1 &
nohup python -u level2_order_predict_future.py -c configs/level2_order_predict_future_config.json > logs/level2_order_predict_future.log 2>&1 &
nohup python -u level3_order_predict_future.py -c configs/level3_order_predict_future_config.json > logs/level3_order_predict_future.log 2>&1 &