import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # 归一化
from keras.layers import Dense, LSTM # # 分别从keras的layers(分层包)中导入全连接层，LSTM层
import keras

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载黄金价格数据
dataset = pd.read_csv('./DataSet/gold_price.csv',index_col=0) # 读取黄金价格数据，将日期列设为索引列
# print(dataset)
dataset = dataset.iloc[:,0] # 只取第一列，即黄金价格
# 将数据转换为DataFrame格式, 方便后续处理
dataset = pd.DataFrame(dataset)
# print(dataset)

# 注意时间顺序不能乱，否则会导致模型训练失败
# 训练集长度为总长度的80%
train_len = int(len(dataset)*0.7) # 训练集长度为总长度的70%

# 训练集
train_data = dataset.iloc[:train_len,:] # 训练集为所有数据的前70%
# 测试集
test_data = dataset.iloc[train_len:,:] # 测试集为所有数据的后30%
print(train_data)

# 归一化
scaler = MinMaxScaler(feature_range=(0,1)) # 归一化到0-1之间
train_data = scaler.fit_transform(train_data) # 训练集归一化
test_data = scaler.fit_transform(test_data) # 测试集归一化

# 划分特征和标签
x_train = []
y_train = []
x_test = []
y_test = []

# 黄金的价格是以五天一个周期做预测，所以每个周期有五天的数据
# 利用for 循环进行训练集的特征和标签的制作，例如用前五天的数据预测第六天的数据(前五天是特征，第六天是标签)
for i in range(5,len(train_data)):
    x_train.append(train_data[i-5:i,0])
    y_train.append(train_data[i,0])

for i in range(5,len(test_data)):
    x_test.append(test_data[i-5:i,0])
    y_test.append(test_data[i,0])

# 转换为numpy数组
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

#循环神经网络的特征需要是3维的，[样本数(训练集长度-5)，时间步长(5)，特征数(1)]，所以需要reshape
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#利用keras搭建神经网络
model = keras.Sequential()
# 第一个LSTM层，80个神经元，返回序列（因为要连接第二个LSTM层），激活函数为relu
model.add(LSTM(80, return_sequences=True, activation='relu'))
# 第二个LSTM层，100个神经元，不返回序列（因为后面要连接全连接层，全连接层只接受2维输入，所以这里不返回每个神经元的输出序列，只返回最后一个时间步的输出序列），激活函数为relu
model.add(LSTM(100, return_sequences=False, activation='relu'))
# 全连接层，10个神经元，激活函数为relu
model.add(Dense(10,activation='relu'))
# 输出层，1个神经元，无激活函数，用于回归任务，输出预测值
model.add(Dense(1))

# 对神经网络编译
# 优化器：Adam，学习率为0.01
# 损失函数：均方误差（mse）
# 评价指标：平均绝对误差（mae）
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss='mse')

# 训练模型
history = model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_data=(x_test,y_test))

# 保存训练好的模型
model.save('./Model/gold_price_model.h5')

# 绘制训练集和验证集的损失函数值变化图
plt.plot(history.history['loss'], label='loss') # 训练集损失函数值
plt.plot(history.history['val_loss'], label='val_loss') # 验证集损失函数值
plt.title('LSTM模型训练集和验证集的损失函数值变化图')
plt.legend() # 显示图例
plt.show() # 显示图像
