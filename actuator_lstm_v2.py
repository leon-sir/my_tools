import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from typing import Tuple, Optional  # noqa: F401
import random  # 导入 random 模块

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self):
        self.using_real_data = True
        self.lr = 1e-4
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


def export_network(model: nn.Module, actuator_network_path: str, config: Config):
    """将模型导出为 TorchScript (.pt) 和 ONNX (.onnx) 格式"""

    # 在CPU上进行导出更安全，不依赖特定硬件
    device_for_export = "cpu"
    model.eval()    # 为了让导出的模型也能重置状态，需要先切换到评估模式
    model.to(device_for_export)
    model_scripted = torch.jit.script(model)
    model_scripted.save(actuator_network_path)
    print(f"Model successfully saved to {actuator_network_path}")


def train_actuator_network(train_x: None, train_y: None,
                           actuator_network_path: str, config: Config):

    model = LSTM_Net(config)
    model = model.to(config.device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           eps=config.eps, weight_decay=config.weight_decay)

    # train_len = train_x.shape[0] - config.num_time_steps -1
    train_len = train_x.shape[0] - config.num_time_steps

    for iter in range(config.iterations):
        # prepare batch_size data
        # start_indices = np.random.randint(0, train_len, size=config.batch_size)
        start_indices = np.random.randint(0, train_len, size=config.batch_size)

        h = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)     # 短期（工作）记忆
        c = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)     # 长期记忆
        model.zero_grad()

        # Truncated Backpropagation Through Time（截断式反向传播TBPTT）
        # 参考 https://lightning.ai/docs/pytorch/stable/common/tbptt.html
        total_loss = 0

        for t in range(config.num_time_steps):
            current_x_batch = train_x[start_indices + t].to(config.device)   # Shape: [batch_size, 1]
            target_y_batch = train_y[start_indices + t].to(config.device)   # Shape: [batch_size, 1]

            current_x_batch = current_x_batch.unsqueeze(1)
            # 将上一步的隐藏状态分离，避免梯度无限回传
            hidden_prev = (h.detach(), c.detach())

            pred, (h, c) = model(current_x_batch, hidden_prev)

            loss = criterion(pred.squeeze(1), target_y_batch)

            # 约后面的信号应该预测越准
            # weight = 0.1 + 0.9 * (t / (config.num_time_steps - 1))

            # loss = loss * weight
            total_loss += loss.item()

            # 反向传播（可以在每个时间步都做，也可以累积total loss在循环外）
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iter % 100 == 0:
            mae = total_loss / config.num_time_steps / config.output_scale**2
            print(f"Iteration: {iter} | loss {loss.item()}| mae: {mae:.4f}")

    # print total loss
    print(f"Finished Training. Final Loss: {loss.item():.6f}")

    export_network(model, actuator_network_path, config)

    return model, hidden_prev


def main():
    # 训练过程
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path of data files")
    parser.add_argument("--output", type=str, required=True, help="Path to save or load the actuator network model")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
    model = torch.jit.load(policy_path).to("cpu")

    # 滑动窗口lstm的隐藏变量每次都要重置，自回归lstm需要保留
    h_0_val = torch.zeros(config.num_layers, 1, config.hidden_size).cpu()
    c_0_val = torch.zeros(config.num_layers, 1, config.hidden_size).cpu()
    hidden_prev = (h_0_val, c_0_val)
    predictions = []

    # online prediction
    for i in range(val_x.shape[0] - config.num_time_steps):
        # 当前时间点的输入
        current_input = val_x[i, :].unsqueeze(0).unsqueeze(0)

        # 进行预测
        pred, hidden_prev = model(current_input, hidden_prev)

        # pred 的形状是 [1, 1, 1]，我们直接取出数值
        predictions.append(pred.item())

    # 准备绘图用的真实值 y
    # 预测是从第 num_time_steps 个点开始的
    true_y_for_plotting = val_y[config.num_time_steps:]
    time_steps_for_plotting = time_steps[config.num_time_steps:]

    plt.plot(time_steps_for_plotting, true_y_for_plotting.ravel(), c='y', label='y true')
    plt.plot(time_steps_for_plotting, predictions,
             c='b', label='y predicted', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()

    export_network(model, policy_path, config)


if __name__ == "__main__":
    main()
