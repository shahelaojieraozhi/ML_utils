import torch
import torch.nn.functional as Fun
import pandas as pd
from sklearn import datasets

dataset = pd.read_excel('E:\\File_cache\\feiq\\Recv Files\\1107李昱-120组数据-特征提取\\data_9.xlsx')
num = 9
data = dataset.iloc[:, :num].values
iris_type = dataset.iloc[:, -2].values
print(data.shape)
print(iris_type.shape)

input = torch.FloatTensor(data)
print(input.shape)
label = torch.LongTensor(iris_type)
print(label.shape)


# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=num, n_hidden=200, n_output=3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss()
# 设定损失函数

for i in range(10000):
    out = net(input)
    loss = loss_func(out, label)
    # 输出与label对比
    optimizer.zero_grad()
    # 初始化
    loss.backward()
    optimizer.step()

out = net(input)
# out是一个计算矩阵
prediction = torch.max(out, 1)[1]

# 实际y输出数据
target_y = label.data.numpy()
print(target_y)
pred_y = prediction.numpy()
# 预测y输出数列
print(pred_y)

# 计算准确率——自己写的
L = pred_y.shape[0]
flag = 0
for i in range(L):
    if pred_y[i] == target_y[i]:
        flag += 1
print("bp准确率为：", flag/L)
