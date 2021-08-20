import numpy as np
from sklearn import datasets
import torch
from sklearn.model_selection import train_test_split

# 数据准备
dataset = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'],
    test_size=0.2
)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(x_train, y_train)
print(x_train.shape)
print(y_train.shape)

# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        # x = Fun.relu(self.hidden(x))  # activation function for hidden layer we choose sigmoid
        x = self.out(x)
        return x


net = Net(n_feature=4, n_hidden=5, n_output=3)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # SGD: 随机梯度下降
# optimizer = torch.optim.SGD(quantized_net.parameters(), lr=0.02)  # SGD: 随机梯度下降
loss_func = torch.nn.CrossEntropyLoss()  # 针对分类问题的损失函数![在这里插入图片描述](https://img-blog.csdnimg.cn/20190108120127973.png)

# 训练数据
for t in range(1000):
    out = net(x_train)  # input x and predict based on x
    loss = loss_func(out, y_train)  # 输出与label对比
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    out = net(x_train)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    # out = quantized_net(input)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    target_y = y_train.data.numpy()
    if t % 100 == 0:
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print("train accuracy:", accuracy)

    out = net(x_test)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    # out = quantized_net(input)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    target_y = y_test.data.numpy()
    if t % 100 == 0:
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print("test accuracy:", accuracy)


w1 = net.hidden.weight.detach().numpy().T
b1 = net.hidden.bias.detach().numpy()
w2 = net.out.weight.detach().numpy().T
b2 = net.out.bias.detach().numpy()

print(w1)
print(b1)
print(w2)
print(b2)

maxmw1 = abs(w1).max()
maxmw2 = abs(w2).max()
maxmb1 = abs(b1).max()
maxmb2 = abs(b2).max()
if (maxmb1 > maxmw1):
    maxmw1 = maxmb1
if (maxmb2 > maxmw2):
    maxmw2 = maxmb2
w1 = w1 / maxmw1
b1 = b1 / maxmw1
w2 = w2 / maxmw2
b2 = b2 / maxmw2

sourceMatrix1 = np.insert(w1, 4, values = b1, axis = 0)
np.savetxt("w1.txt", sourceMatrix1)
sourceMatrix2 = np.insert(w2, 4, values = b2, axis = 0)
np.savetxt("w2.txt", sourceMatrix2)