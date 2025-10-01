import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from typing import Tuple, Optional 


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

""" To do list
晚点把.csv的数据步长改为0.005s
"""


class Config:
    def __init__(self):
        self.using_real_data = False
        self.lr = 1e-2
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.epochs = 200
        # self.batch_size = 200
        self.batch_size = 1
        self.num_time_steps = 50
        self.device = "cuda:0"
        self.in_dim = 1
        self.units = 32
        self.num_layers = 2
        self.out_dim = 1
        self.act = "softsign"
        self.dt = 0.001
        self.iterations = 6000
        self.hidden_size = 16


def load_data(data_path):
    data = pd.read_csv(data_path)
    if len(data) < 1:
        return None, 0

    num_actuators = sum(1 for col in data.columns if col.startswith("w_f"))
    columns = ["w_f", "F", "N1", "N2", "P3", "T5"]

    data_dict = {"Time": []}
    for col in columns:
        data_dict[col] = []

    # 单独处理Time列（只有一列，没有数字后缀）
    data_dict["Time"].append(data["Time"].values)

    # data_dict = {col: [] for col in columns}
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

        self.rnn = nn.LSTM(
            input_size=config.in_dim,    # feature_len = 1
            hidden_size=config.hidden_size,  # 隐藏记忆单元个数hidden_len = 16
            num_layers=config.num_layers,    # 网络层数 = 1 or 2
            batch_first=True,         # 在传入数据时,按照[batch,seq_len,feature_len]的格式
        )

        for p in self.rnn.parameters():  # 对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(config.hidden_size, config.out_dim)  # 输出层

    def forward(self, x: torch.Tensor, hidden_prev: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
        x: 一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev: 第一个时刻空间上所有层的记忆单元(batch, num_layer, hidden_len)
        输出out(batch,seq_len,hidden_len) 和 hidden_prev(batch,num_layer,hidden_len)
        """

        out, hidden_prev = self.rnn(x, hidden_prev)     # out hn

        '''
        out的最后一维输出等于hn
        '''
        # 因为要把输出传给线性层处理，这里将batch和seq_len维度打平
        # 再把batch=1添加到最前面的维度（为了和y做MSE）
        # [batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = out.view(-1, self.hidden_size)

        # [seq_len,hidden_len]->[seq_len,output_size=1]
        out = self.linear(out)
        # [seq_len,output_size=1]->[batch=1,seq_len,output_size=1]
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


def train_actuator_network(train_x: None, train_y: None,
                           actuator_network_path: str, config: Config):

    model = LSTM_Net(config)
    model = model.to(config.device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           eps=config.eps, weight_decay=config.weight_decay)
    # 初始化记忆单元h0[batch,num_layer,hidden_len]
    # 初始化记忆单元h0[batch,num_layer,hidden_len]
    h_0 = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)
    c_0 = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).to(config.device)
    hidden_prev = (h_0, c_0)
    # hidden_prev = torch.zeros(config.batch_size, config.num_layers, config.hidden_size).to(config.device)
    train_len = train_x.shape[0] - config.num_time_steps
    num_time_steps = config.num_time_steps

    for iter in range(config.iterations):
        if config.using_real_data:
            start_idx = np.random.randint(0, train_len)
            end_idx = start_idx + config.num_time_steps

            # [seq_len, feature] -> [batch=1, seq_len, feature]
            x = train_x[start_idx:end_idx-1].unsqueeze(0).to(config.device)
            y = train_y[start_idx+1:end_idx].unsqueeze(0).to(config.device)
            """ To do list
            涡喷数据最好再加个normalizer
            """
            pass
        else:
            # 在0~3之间随机取开始的时刻点
            start = np.random.randint(3, size=1)[0]
            # 在[start,start+10]区间均匀地取num_points个点
            time_steps = np.linspace(start, start + 10, num_time_steps)
            data = np.sin(time_steps)
            # [num_time_steps,] -> [num_points,1]
            data = data.reshape(num_time_steps, 1)
            # 输入前49个点(seq_len=49)，即下标0~48 [batch, seq_len, feature_len]  # [ batch_size, seq_len, input_size]
            x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1).to(config.device)
            # 预测后49个点，即下标1~49
            y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1).to(config.device)
            # 以上步骤生成(x,y)数据对

        '''
            因为结点的个数是16，所以词向量的维度为16，所以初始输入的向量h0的维度也需要16
        '''
        output, hidden_prev = model(x, hidden_prev)  # 喂入模型得到输出
        '''
        对预测结果进行传递，因为时间是一直走的 s(0)->[1,2]->[2,3]，所以上一次的结果需要传到下一次
        h0维度是(num_layers * directions, batch_size, hidden_dim)
        hn的维度是(num_layers * directions, batch_size, hidden_dim)
        output的维度是(seq_len, batch_size, hidden_dim * directions)
        '''
        hidden_prev = (hidden_prev[0].detach(), hidden_prev[1].detach())
        # hidden_prev = hidden_prev.detach()  # 或着hidden_prev.data
        # print("hidden_prev_detach", hidden_prev)
        loss = criterion(output, y)  # 计算MSE损失
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
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
config = Config()

data_dict, num_jets = load_data(data_path)

if config.using_real_data:
    xs = torch.tensor(data_dict["w_f"][:, 0:1], dtype=torch.float)
    ys = torch.tensor(data_dict["F"][:, 0:1], dtype=torch.float)

    num_data = xs.shape[0]
    num_train = int(num_data * 0.8)

    # 训练数据
    train_x = xs[:num_train]
    train_y = ys[:num_train]

    # 验证数据
    val_x = xs[num_train:]
    val_y = ys[num_train:]

    # xs.append(data_dict["w_f"])
    # xs = torch.tensor(xs, dtype=torch.float)
    # ys.append(data_dict["F"])
    # ys = torch.tensor(ys, dtype=torch.float)
    # num_data = xs.shape[0]
    # num_train = num_data // 5 * 4
    # num_test = num_data - num_train
    # 分割数据
    model, hidden_prev = train_actuator_network(train_x=train_x, train_y=train_y,
                                                actuator_network_path=output_path, config=config)
else:
    model, hidden_prev = train_actuator_network(train_x=torch.tensor([0]), train_y=0, actuator_network_path=output_path, config=config)


# Validation
if config.using_real_data:
    pass
else:
    # 先用同样的方式生成一组数据x,y
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start + 10, config.num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(config.num_time_steps, 1)
    val_x = torch.tensor(data[:-1]).float().view(1, config.num_time_steps - 1, 1)
    val_y = torch.tensor(data[1:]).float().view(1, config.num_time_steps - 1, 1)

model.cpu()


# hidden_prev = hidden_prev.cpu()
# hidden_prev = torch.zeros(config.batch_size, config.num_layers, config.hidden_size).cpu()
h_0_val = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).cpu()
c_0_val = torch.zeros(config.num_layers, config.batch_size, config.hidden_size).cpu()
hidden_prev = (h_0_val, c_0_val)
predictions = []

input = val_x[:, 0, :]           # 取seq_len里面第0号数据
input = input.view(1, 1, 1)  # input：[1,1,1]
for _ in range(val_x.shape[1]):  # 迭代seq_len次
    pred, hidden_prev = model(input, hidden_prev)
    # 预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

val_x = val_x.data.numpy()
val_y = val_y.data.numpy()
plt.plot(time_steps[:-1], val_x.ravel(), label='x values (line)')

plt.scatter(time_steps[:-1], val_x.ravel(), c='r', label='x (start)')  # x值
plt.scatter(time_steps[1:], val_y.ravel(), c='y', label='y true')  # y值
plt.scatter(time_steps[1:], predictions, c='b', label='y predicted')  # y的预测值
plt.legend
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