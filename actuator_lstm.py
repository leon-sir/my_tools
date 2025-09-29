import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


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
        self.num_layers = 1
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

        self.rnn = nn.RNN(
            input_size=config.in_dim,    # feature_len = 1
            hidden_size=config.hidden_size,  # 隐藏记忆单元个数hidden_len = 16
            num_layers=config.num_layers,    # 网络层数 = 1
            batch_first=True,         # 在传入数据时,按照[batch,seq_len,feature_len]的格式
        )

        for p in self.rnn.parameters():  # 对RNN层的参数做初始化
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(config.hidden_size, config.out_dim)  # 输出层

    def forward(self, x, hidden_prev):
        """
        x：一次性输入所有样本所有时刻的值(batch,seq_len,feature_len)
        hidden_prev：第一个时刻空间上所有层的记忆单元(batch, num_layer, hidden_len)
        输出out(batch,seq_len,hidden_len) 和 hidden_prev(batch,num_layer,hidden_len)
        """
        # print("x", x)
        # print("hidden_prev_1", hidden_prev)
        out, hidden_prev = self.rnn(x, hidden_prev)     # out hn
        # print("hidden_prev_2", hidden_prev)
        # print("out",out)
        '''
        out的最后一维输出等于hn
        '''
        # print(out.shape)  # [49, 16]，49个点49个输出一次h0-h49
        # 因为要把输出传给线性层处理，这里将batch和seq_len维度打平
        # 再把batch=1添加到最前面的维度（为了和y做MSE）
        # [batch=1,seq_len,hidden_len]->[seq_len,hidden_len]
        out = out.view(-1, self.hidden_size)

        # [seq_len,hidden_len]->[seq_len,output_size=1]
        out = self.linear(out)
        # [seq_len,output_size=1]->[batch=1,seq_len,output_size=1]
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


def train_actuator_network(xs: None, ys: None,
                           actuator_network_path: str, config: Config):

    model = LSTM_Net(config)
    model = model.to(config.device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           eps=config.eps, weight_decay=config.weight_decay)
    # 初始化记忆单元h0[batch,num_layer,hidden_len]
    hidden_prev = torch.zeros(config.batch_size, config.num_layers, config.hidden_size).to(config.device)
    num_time_steps = config.num_time_steps

    for iter in range(config.iterations):
        if config.using_real_data and xs is not None:
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
        hidden_prev = hidden_prev.detach()  # 或着hidden_prev.data
        # print("hidden_prev_detach", hidden_prev)
        loss = criterion(output, y)  # 计算MSE损失
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))

    # print total loss
    print("Iteration: {} loss {}".format(iter, loss.item()))
    # iter == config.iterations
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(actuator_network_path)  # Save

    print("****\n Exported actuator lstm networks successfully\n *****")

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
    model, hidden_prev = train_actuator_network(xs=1, ys=1, actuator_network_path=output_path, config=config)
else:
    model, hidden_prev = train_actuator_network(xs=None, ys=None, actuator_network_path=output_path, config=config)

# 测试过程
# 先用同样的方式生成一组数据x,y
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, config.num_time_steps)
data = np.sin(time_steps)
data = data.reshape(config.num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, config.num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, config.num_time_steps - 1, 1)

model.cpu()
"""
为什么隐藏变量是否初始化差这么多？
"""

# hidden_prev = hidden_prev.cpu()
hidden_prev = torch.zeros(config.batch_size, config.num_layers, config.hidden_size).cpu()
predictions = []

input = x[:, 0, :]           # 取seq_len里面第0号数据
input = input.view(1, 1, 1)  # input：[1,1,1]
for _ in range(x.shape[1]):  # 迭代seq_len次
    pred, hidden_prev = model(input, hidden_prev)
    # 预测出的(下一个点的)序列pred当成输入(或者直接写成input, hidden_prev = model(input, hidden_prev))
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy()
y = y.data.numpy()
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[:-1], x.ravel(), c='r')  # x值
plt.scatter(time_steps[1:], y.ravel(), c='y')  # y值
plt.scatter(time_steps[1:], predictions, c='b')  # y的预测值
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