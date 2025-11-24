import numpy as np # 导入numpy库
import pandas as pd # 导入pandas库
import matplotlib.pyplot as plt # 导入matplotlib库
from pandas import read_excel
from sklearn.preprocessing import MinMaxScaler # 导入sklearn库中的线性回归模型
from sklearn.model_selection import train_test_split # 导入sklearn库中的数据集划分函数
import keras # 导入keras库
from keras.layers import Dense # 导入keras库中的全连接层
from keras.utils.np_utils import to_categorical #导入keras库中的one-hot编码函数
from sklearn.metrics import classification_report # 导入sklearn库中的分类报告函数

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
dataset = read_excel('./DataSet/乳腺癌原始数据.xlsx')
# print(dataset)

# 提取特征
x = dataset.iloc[:,1 :-1]  # 提取所有行，除了最后一列
# print(x)

# 提取标签
y = dataset.iloc[:,-1]
print(y)

# 划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 将标签转化为独热编码(one-hot格式)，因为在多分类问题中，神经网络无法直接识别类别，容易出现1 + 2 = 3，类别混乱
# 所以需要将标签转化为独热编码，例如：0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train_one = to_categorical(y_train,2) # 2表示有2个类别
y_val_one = to_categorical(y_val,2)

# 将数据特征归一化
sc = MinMaxScaler(feature_range=(0,1)) # 归一化到0-1之间
x_train = sc.fit_transform(x_train) # 训练集归一化
x_val = sc.fit_transform(x_val) # 验证集归一化

# 利用keras构建模型，Dense表示全连接神经网络
model = keras.Sequential()
# 矩阵乘法的解释：这里五个特征，所以输入神经元的数量(输入)为5，所以特征向量x为5*1
# 又因为隐藏层神经元数量为10，所以隐藏层的权重矩阵W设置为10*5，
# 一层的全连接层本质上就是将隐藏层的权重矩阵与输入层的特征向量相乘，再加上隐藏层的偏置(b)向量（隐藏层的偏置向量为10*1），得到隐藏层的输出向量为10*1
# 矩阵乘法: W * x + b = z，其中W为隐藏层的权重矩阵，x为输入层的特征向量，b为隐藏层的偏置向量，z为隐藏层的输出向量
model.add(Dense(10,activation='relu',input_dim=5)) # (第一层隐藏层)输入层到隐藏层，10表示该隐藏层神经元的数量，activation表示激活函数
model.add(Dense(10,activation='relu')) # (第二层隐藏层)隐藏层到隐藏层，10表示该隐藏层神经元的数量，activation表示激活函数
model.add(Dense(2,activation='softmax')) # (输出层)隐藏层到输出层，2表示输出层神经元的数量，activation表示激活函数

# 对神经网络进行编译
# 编译模型，loss表示损失函数(这里用的是交叉熵损失函数)，optimizer表示参数更新优化器(这里使用SGD，随机梯度下降法)，metrics表示评估指标
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 训练模型，epochs表示训练轮数，batch_size表示每次训练的样本数量，verbose表示训练过程中控制台是否显示信息，把验证集当作验证集验证是否过拟合
# 过拟合：验证集和训练集的损失函数值相差较大，训练集(平时练习)好，但验证集(月考)差
history = model.fit(x_train, y_train_one, epochs=100, batch_size=15, verbose=1,validation_data=(x_val, y_val_one))

# 保存模型信息，后缀h5是keras训练出来的模型文件
model.save('./Model/breast_cancer_model.h5')

# 绘制训练集和验证集的损失函数值变化图
plt.plot(history.history['loss'], label='loss') # 训练集损失函数值
plt.plot(history.history['val_loss'], label='val_loss') # 验证集损失函数值
plt.title('全连接神经网络模型损失函数值变化图')
plt.ylabel('损失函数值')
plt.xlabel('训练轮数')
plt.legend(['训练集', '验证集'], loc='upper right') # 显示图例，loc表示图例位置
plt.show()

# 绘制训练集和验证集的准确率变化图
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('全连接神经网络模型准确率变化图')
plt.ylabel('准确率')
plt.xlabel('训练轮数')
plt.legend(['训练集', '验证集'], loc='upper right') # 显示图例，loc表示图例位置
plt.show()



