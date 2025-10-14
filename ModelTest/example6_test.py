import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载黄金价格数据
dataset = pd.read_csv('../DataSet/gold_price.csv',index_col=0) # 读取黄金价格数据，将日期列设为索引列
# print(dataset)
dataset = dataset.iloc[:,0] # 只取第一列，即黄金价格
# 将数据转换为DataFrame格式, 方便后续处理
dataset = pd.DataFrame(dataset)
# print(dataset)

# 注意时间顺序不能乱，否则会导致模型训练失败
# 训练集长度为总长度的80%
train_len = int(len(dataset)*0.7) # 训练集长度为总长度的70%

# 测试集
test_data = dataset.iloc[train_len:,:] # 测试集为所有数据的后30%

# 归一化
scaler = MinMaxScaler(feature_range=(0,1)) # 归一化到0-1之间
test_data = scaler.fit_transform(test_data) # 测试集归一化
# print(test_data)

# 划分特征和标签
x_test = []
y_test = []

# 黄金的价格是以五天一个周期做预测，所以每个周期有五天的数据
# 利用for 循环进行训练集的特征和标签的制作，例如用前五天的数据预测第六天的数据(前五天是特征，第六天是标签)
for i in range(5,len(test_data)):
    x_test.append(test_data[i-5:i,0])
    y_test.append(test_data[i,0])

# 转换为numpy数组
x_test, y_test = np.array(x_test), np.array(y_test)

#循环神经网络的特征需要是3维的，[样本数(训练集长度-5)，时间步长(5)，特征数(1)]，所以需要reshape
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#导入模型
model = load_model('../Model/gold_price_model.h5')

#利用模型来测试
predicted = model.predict(x_test)
# print(len(predicted))

# 反归一化,注意归一化和反归一化维度要一致，这里是(样本数,1)
predictions = scaler.inverse_transform(predicted)
y_test = pd.DataFrame(y_test) # 转换为DataFrame格式, 二维化
real_y = scaler.inverse_transform(y_test) # 因为inverse_transform要求输入是二维的，所以需要转换一下
# print(real_y,predictions)

# 打印模型的评价指标
rmse = sqrt(mean_squared_error(predictions, real_y)) # 计算均方根误差
mape = np.mean(np.abs((predictions - real_y) / predictions)) # 计算平均绝对百分比误差
print(f'均方根误差(RMSE): {rmse:.4f}')
print(f'平均绝对百分比误差(MAPE): {mape:.4f}')

# 可视化预测结果
plt.figure(figsize=(12,6))
plt.plot(real_y,label='真实值')
plt.plot(predictions,label='预测值')
plt.title('黄金价格预测')
plt.xlabel('时间')
plt.ylabel('黄金价格')
plt.legend()
plt.show()
