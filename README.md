# install
~bash$ 

pip3 install torch --index-url https://download.pytorch.org/whl/cu118 pip3 install numpy==1.26.4 pandas==2.1.3
conda install matplotlib pandas

Hasee wsl 虚拟机：
python>= 3.8
pip3 install torch==2.8.0 torchvision
pip install numpy==1.23.5 pandas==2.3.1
（台式cu121）

# example
~bash$ 
<!-- python3 actuator_net_mlp.py --mode train --data train_motor_csv/test.csv --output train_motor_csv/motor_test.pt -->
python3 actuator_net_mlp.py --mode train --data train_motor_csv/robot_data_04_11_13_24_Ankle.csv --output train_motor_csv/motor_Ankle.pt

# for ymbot
~bash$ 

python3 actuator_net_mlp.py --mode train --data train_motor_csv/robot_data_04_11_13_24_Ankle.csv --output train_motor_csv/motor_Ankle.pt

python3 actuator_net_mlp.py --mode train --data train_motor_csv/robot_data_04_11_13_24_HipKnee.csv --output train_motor_csv/motor_HipKnee.pt

python3 actuator_net_mlp.py --mode train --data train_motor_csv/robot_data_04_11_13_24.csv --output train_motor_csv/motor_all.pt

# after training change mode to play

python3 actuator_net_mlp.py --mode play --data train_motor_csv/robot_data_04_11_13_24_Ankle.csv --output train_motor_csv/motor_Ankle.pt

python3 actuator_net_mlp.py --mode play --data train_motor_csv/robot_data_04_11_13_24_HipKnee.csv --output train_motor_csv/motor_HipKnee.pt

python3 actuator_net_mlp.py --mode play --data train_motor_csv/robot_data_04_11_13_24.csv --output train_motor_csv/motor_all.pt

# test lstm

```bash
python3 test_lstm.py
python3 actuator_lstm.py --data train_turbo_csv/fly_robot_LPV_data.csv --output train_turbo_csv
python3 actuator_rnn.py --data train_turbo_csv/fly_robot_LPV_data.csv --output train_turbo_csv/fly_robot_LPV.pt
```


