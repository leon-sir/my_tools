install
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 pip3 install numpy==1.26.4 pandas==2.1.3

example
python3 actuator_net_mlp.py --mode train --data train_motor_csv/test.csv --output train_motor_csv/motor.pt
