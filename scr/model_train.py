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
