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

# 导入数据集
dataset = pd.read_csv('./DataSet/AirQuality_ShiJiaZhuang.csv')
#print(dataset)

# 因为是回归任务，所以不需要one-hot编码，且应该先归一化再划分数据集，避免测试集和训练集的归一化比例不一致
# 归一化的公式：x = (x - min) / (max - min)
sc = MinMaxScaler(feature_range=(0,1))
#将**所有**数据归一化，包括特征和标签
vaild_dataset = dataset.drop(columns=['日期','质量等级'])
data_sum = sc.fit_transform(vaild_dataset)

# 将归一化好的数据转化为dataframe(表格)格式，方便后续处理
data_sum = pd.DataFrame(data_sum)
# print(data_sum)

# 划分数据集
x = data_sum.iloc[:, 1:]
y = data_sum[0]
# print(x);print(y)

# 划分数据集合
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 构建全连接神经网络模型
model = keras.Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))

# 编译模型，更新参数方式，用哪些损失函数，用什么评价指标
model.compile(optimizer='SGD',loss='mse',metrics=['mae'])

# 训练模型,并将训练过程中的损失值和评价指标值保存下来
# batch_size: 每次训练的样本数量
# epochs: 训练的轮数
# validation_data: 验证集
# verbose: 训练过程中是否打印日志
history = model.fit(x_train, y_train, epochs=120, batch_size=24, validation_data=(x_test, y_test), verbose=1)

# 保存模型
model.save('./Model/air_quality_model.h5')

# 绘制训练集和验证集的损失函数值变化图
plt.plot(history.history['loss'], label='loss') # 训练集损失函数值
plt.plot(history.history['val_loss'], label='val_loss') # 验证集损失函数值
plt.title('全连接神经网络模型损失函数值变化图')
plt.ylabel('损失函数值')
plt.xlabel('训练轮数')
plt.legend(['训练集', '验证集'], loc='upper right') # 显示图例，loc表示图例位置
plt.show()

# 注：回回归任务没有准确率的概念，只有均方根误差，均方误差等评价指标


