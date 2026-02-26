import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# 解决画图里中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("data\\breast_cancer_data.csv")
# pd里的read_csv函数可以直接读取csv文件，并将其转换为DataFrame格式，方便后续的数据处理和分析。
# print(dataset)

# 提取数据中的特征
X = dataset.iloc[:, :-1]
# 读取数据集中的所有行和除最后一列以外的所有列作为特征数据，存储在变量X中。
# print(X)

# 提取数据集中的标签
Y = dataset['target']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#将20%的数据用作测试集，剩余的80%用作训练集。random_state随机种子参数设置为42，确保每次运行代码时划分结果相同，便于结果的复现和比较。


# 将数据标签转化为one——hot向量格式
y_train_one = to_categorical(y_train, 2)# 2种分类，0和1，所以设置为2
y_test_one = to_categorical(y_test, 2)

# 将数据特征进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)# fit_transform方法首先计算训练数据的最小值和最大值，然后将训练数据进行归一化处理，使其缩放到指定的范围内（0到1）。fit_transform方法返回归一化后的训练数据，并将其存储在变量x_train中。
x_test = sc.transform(x_test)#不fit！


# 利用Keras框架帮助搭建深度学习网络模型
model = keras.Sequential()# Sequential模型是Keras中最常用的模型类型，它表示一个线性的层次结构，可以通过add方法逐层添加神经网络层。
model.add(Dense(10, activation='relu'))# 添加一个全连接层，包含10个神经元，激活函数为ReLU。
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))# 添加一个全连接层，包含2个神经元，激活函数为softmax。这个层用于输出分类结果，因为我们有两个类别（0和1），所以设置为2。softmax激活函数将输出转换为概率分布，使得输出的值在0到1之间，并且所有输出的和为1，适用于多分类问题。

# 对神经网络进行编译
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])# 损失函数选择categorical_crossentropy，优化器选择SGD，评估指标选择accuracy。

history = model.fit(x_train, y_train_one, epochs=150, batch_size=64, verbose=2, validation_data=(x_test, y_test_one))
# 训练模型，使用训练数据x_train和y_train_one进行训练，设置训练轮数为150，批次大小为64，verbose=2表示在训练过程中显示每个epoch的进度信息。validation_data参数指定了验证集的数据，用于在每个epoch结束后评估模型的性能。
# 保存路径
import os
if not os.path.exists('output'):
    os.makedirs('output')

model.save('output/model.h5')# 将训练好的模型保存为model.h5文件


# 绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()


# 绘制训练集和验证集的准确率的对比图
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("全连接神经网络accuracy值图")
plt.legend()
plt.show()









# 在pytorch框架下搭建全连接神经网络模型的代码实现如下：
# #import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader

# # 解决画图里中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 加载数据集
# dataset = pd.read_csv("data\\breast_cancer_data.csv")

# # 提取数据中的特征
# X = dataset.iloc[:, :-1]
# # 提取数据集中的标签
# Y = dataset['target']

# # 划分训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # 将数据特征进行归一化
# sc = MinMaxScaler(feature_range=(0, 1))
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# # 转换为Tensor
# x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
# x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# # 构建DataLoader
# train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64)

# # 定义模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(x_train.shape[1], 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 2)
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model = Net()

# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 训练模型
# num_epochs = 150
# train_loss_list = []
# val_loss_list = []
# train_acc_list = []
# val_acc_list = []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for xb, yb in train_loader:
#         optimizer.zero_grad()
#         outputs = model(xb)
#         loss = criterion(outputs, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == yb).sum().item()
#         total += yb.size(0)
#     train_loss = running_loss / total
#     train_acc = correct / total
#     train_loss_list.append(train_loss)
#     train_acc_list.append(train_acc)

#     # 验证集
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             outputs = model(xb)
#             loss = criterion(outputs, yb)
#             val_loss += loss.item() * xb.size(0)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == yb).sum().item()
#             val_total += yb.size(0)
#     val_loss = val_loss / val_total
#     val_acc = val_correct / val_total
#     val_loss_list.append(val_loss)
#     val_acc_list.append(val_acc)

#     if (epoch+1) % 10 == 0 or epoch == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}] "
#               f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
#               f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

# # 保存模型
# import os
# if not os.path.exists('output'):
#     os.makedirs('output')
# torch.save(model.state_dict(), 'output/model.pth')

# # 绘制训练集和验证集的loss值对比
# plt.plot(train_loss_list, label='train')
# plt.plot(val_loss_list, label='val')
# plt.title("全连接神经网络loss值图")
# plt.legend()
# plt.show()

# # 绘制训练集和验证集的准确率的对比图
# plt.plot(train_acc_list, label='train')
# plt.plot(val_acc_list, label='val')
# plt.title("全连接神经网络accuracy值图")
# plt.legend()
# plt.show()

# # 分类报告
# model.eval()
# y_pred = []
# with torch.no_grad():
#     for xb, _ in test_loader:
#         outputs = model(xb)
#         _, predicted = torch.max(outputs, 1)
#         y_pred.extend(predicted.cpu().numpy())
# print(classification_report(y_test, y_pred))