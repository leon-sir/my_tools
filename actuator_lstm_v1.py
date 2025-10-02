import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from typing import Tuple, Optional  # noqa: F401


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

""" To do list
晚点把.csv的数据步长改为0.005s
"""


class Config:
    def __init__(self):
        self.using_real_data = True
        self.lr = 1e-3
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.batch_size = 40
        self.num_time_steps = 50
        self.device = "cuda:0"
        self.in_dim = 1
        self.num_layers = 2     # lstm layers
        self.out_dim = 1
        self.act = "softsign"
        self.dt = 0.005
        self.iterations = 30000
        self.hidden_size = 50
        self.linear_units = 8
        self.input_scale = 1.5  # turbojet fuel flow(W(kg/s))
        self.output_scale = 70  # turbojet force


def load_data(data_path):
    data = pd.read_csv(data_path)
    if len(data) < 1:
        return None, 0

    num_actuators = sum(1 for col in data.columns if col.startswith("w_f"))
    columns = ["w_f", "F", "N1", "N2", "P3", "T5"]

    data_dict = {"Time": []}
    for col in columns:
        data_dict[col] = []

    data_dict["Time"].append(data["Time"].values)

    for col in columns:
        for i in range(num_actuators):
            data_dict[col].append(data[f"{col}_{i}"].values)

    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key]).T

    """ To do list
        多涡喷数据时 data_dict 应该只有一行时间, Time列单独处理
    """
    return data_dict, num_actuators


class LSTM_Net(nn.Module):
    def __init__(self, config=Config):
        super().__init__()

        self.hidden_size = config.hidden_size

        # 注册缩放因子作为模型的缓冲区
        self.register_buffer('input_scale', torch.tensor(config.input_scale, dtype=torch.float))
        self.register_buffer('output_scale', torch.tensor(config.output_scale, dtype=torch.float))

        self.rnn = nn.LSTM(
            input_size=config.in_dim,    # feature_len = 1
            hidden_size=config.hidden_size,  # 隐藏记忆单元个数hidden_len = 16
            num_layers=config.num_layers,    # 网络层数 = 1 or 2
            batch_first=True,         # 在传入数据时,按照[batch,seq_len,feature_len]的格式
        )

        for p in self.rnn.parameters():  # 对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

        """先不加mlp"""
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.linear_units),
            nn.Softsign(),
            # nn.ReLU(),
            nn.Linear(config.linear_units, config.out_dim)
        )
        # self.linear = nn.Linear(config.hidden_size, config.out_dim)  # 输出层

    def forward(self, x: torch.Tensor, hidden_prev: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
        x: 一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev: 第一个时刻空间上所有层的记忆单元(batch, num_layer, hidden_len)
        输出out(batch,seq_len,hidden_len) 和 hidden_prev(batch,num_layer,hidden_len)
        """
        # 1. 应用输入缩放
        x = x / self.input_scale

        out, hidden_prev = self.rnn(x, hidden_prev)     # out hn

        # 因为我们只关心最后一个时间步的输出，所以只对它进行处理
        # out shape: [batch, seq_len, hidden_size]
        last_time_step_out = out[:, -1, :]  # Shape: [batch, hidden_size]

        # 直接将最后一个时间步的输出送入线性层
        # Shape: [batch, out_dim]
        prediction = self.linear(last_time_step_out)

        # 2. 应用输出缩放
        prediction = prediction * self.output_scale
        return prediction.unsqueeze(1), hidden_prev


def train_actuator_network(train_x: None, train_y: None,
                           actuator_network_path: str, config: Config):

    model = LSTM_Net(config)
    model = model.to(config.device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           eps=config.eps, weight_decay=config.weight_decay)

    train_len = train_x.shape[0] - config.num_time_steps

    for iter in range(config.iterations):
        # prepare batch_size data
        start_indices = np.random.randint(0, train_len, size=config.batch_size)

        batch_x = []
        batch_y = []

        for start_idx in start_indices:
            window_end_idx = start_idx + config.num_time_steps
            target_idx = window_end_idx

            # 收集一个窗口的输入和一个目标输出
            batch_x.append(train_x[start_idx:window_end_idx])
            batch_y.append(train_y[target_idx])

        # 4. 将数据列表堆叠成一个批次张量，并移动到GPU
        # x shape: [batch_size, num_time_steps, in_dim] -> [10, 50, 1]
        x = torch.stack(batch_x).to(config.device)
        # y shape: [batch_size, out_dim] -> [10, 1]
        y = torch.stack(batch_y).to(config.device)

        # 初始化记忆单元h0，c0永远是 (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)
        c_0 = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)
        hidden_prev = (h_0, c_0)

        # start_idx = np.random.randint(0, train_len)
        # # end_idx = start_idx + config.num_time_steps
        # # 定义输入窗口和目标点
        # window_end_idx = start_idx + config.num_time_steps
        # target_idx = window_end_idx     # 目标是窗口结束后的那个点

        # # [seq_len, feature] -> [batch=1, seq_len, feature]
        # x = train_x[start_idx:window_end_idx].unsqueeze(0).to(config.device)        # x是包含过去N个点的序列
        # y = train_y[target_idx].unsqueeze(0).to(config.device)                      # y是序列结束后的单个点

        output, _ = model(x, hidden_prev)  # 喂入模型得到输出
        # hidden_prev = (hidden_prev[0].detach(), hidden_prev[1].detach())
        last_output = output.squeeze(1)     # 只取序列的最后一个输出用于计算损失
        # last_output = output[:, -1, :]      # 只取序列的最后一个输出用于计算损失

        # loss = criterion(output, y)  # 计算MSE损失
        loss = criterion(last_output, y)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iter % 1000 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))

    # print total loss
    print(f"Finished Training. Final Loss: {loss.item():.6f}")
    # iter == config.iterations
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(actuator_network_path)  # Save

    print("****\n Exported actuator lstm networks successfully\n*****")

    return model, hidden_prev


# # 训练过程
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path of data files")
parser.add_argument("--output", type=str, required=True, help="Path to save or load the actuator network model")
args = parser.parse_args()
data_path = os.path.join(BASE_PATH, args.data)
output_path = os.path.join(BASE_PATH, args.output)
current_script_name = os.path.splitext(os.path.basename(__file__))[0]
policy_path = os.path.join(output_path, f"{current_script_name}_net.pt")
figure_path = os.path.join(output_path, f"{current_script_name}_figure.png")
config = Config()

data_dict, num_jets = load_data(data_path)

xs = torch.tensor(data_dict["w_f"][:, 0:1], dtype=torch.float)
ys = torch.tensor(data_dict["F"][:, 0:1], dtype=torch.float)
Time = torch.tensor(data_dict["Time"][:, 0:1], dtype=torch.float)

num_data = xs.shape[0]
num_train = int(num_data * 0.8)

# 训练数据
train_x = xs[:num_train]
train_y = ys[:num_train]

# 验证数据
val_x = xs[num_train:]
val_y = ys[num_train:]
time_steps = Time[num_train:]-Time[num_train]

model, hidden_prev = train_actuator_network(train_x=train_x, train_y=train_y,
                                            actuator_network_path=policy_path, config=config)


# Validation
model.cpu()

# 滑动窗口lstm的隐藏变量每次都要重置，自回归lstm需要保留
h_0_val = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).cpu()
c_0_val = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).cpu()
hidden_prev = (h_0_val, c_0_val)
predictions = []

# 遍历验证集，使用滑动窗口进行预测
for i in range(val_x.shape[0] - config.num_time_steps):
    # 构造输入窗口
    start_idx = i
    end_idx = i + config.num_time_steps
    input_window = val_x[start_idx:end_idx].unsqueeze(0) # Shape: [1, num_time_steps, 1]

    # 初始化隐藏状态
    h_0_val = torch.zeros(config.num_layers, 1, config.hidden_size)
    c_0_val = torch.zeros(config.num_layers, 1, config.hidden_size)
    hidden_prev = (h_0_val, c_0_val)

    # 进行预测
    pred, hidden_prev = model(input_window, hidden_prev)

    # pred 的形状是 [1, 1, 1]，我们直接取出数值
    predictions.append(pred.item())


# 准备绘图用的真实值 y
# 预测是从第 num_time_steps 个点开始的
true_y_for_plotting = val_y[config.num_time_steps:]
time_steps_for_plotting = time_steps[config.num_time_steps:]


plt.plot(time_steps_for_plotting, true_y_for_plotting.ravel(), c='y', label='y true')
plt.plot(time_steps_for_plotting + config.num_time_steps*config.dt, predictions, c='b', label='y predicted', linestyle='--')
plt.legend()
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.show()



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, required=True, choices=["train", "play"], help="Choose whether to train or evaluate the actuator network")
#     parser.add_argument("--data", type=str, required=True, help="Path of data files")
#     parser.add_argument("--output", type=str, required=True, help="Path to save or load the actuator network model")

#     args = parser.parse_args()

#     data_path = os.path.join(BASE_PATH, args.data)
#     output_path = os.path.join(BASE_PATH, args.output)

#     config = Config()

#     if args.mode == "train":
#         load_pretrained_model = False
#     elif args.mode == "play":
#         load_pretrained_model = True

#     train_actuator_network_and_plot_predictions(
#         data_path=data_path,
#         actuator_network_path=output_path,
#         load_pretrained_model=load_pretrained_model,
#         config=config,
#     )

# if __name__ == "__main__":
#     main()