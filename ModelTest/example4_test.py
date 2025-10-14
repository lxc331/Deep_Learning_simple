import numpy as np # 导入numpy库
import pandas as pd # 导入pandas库
import matplotlib.pyplot as plt # 导入matplotlib库
from pandas import read_excel
from sklearn.preprocessing import MinMaxScaler # 导入sklearn库中的线性回归模型
from sklearn.model_selection import train_test_split # 导入sklearn库中的数据集划分函数
import keras # 导入keras库
from keras.layers import Dense # 导入keras库中的全连接层
from keras.models import load_model # 导入keras库中的模型加载函数
from math import sqrt #用来求均方根误差
from numpy import concatenate # 用来反归一化
from sklearn.metrics import mean_squared_error # 用来求均方误差

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
dataset = pd.read_csv('../DataSet/AirQuality_ShiJiaZhuang.csv')
# print(dataset)

# 因为是回归任务，所以不需要one-hot编码，且应该先归一化再划分数据集，避免测试集和训练集的归一化比例不一致
# 归一化的公式：x = (x - min) / (max - min)
sc = MinMaxScaler(feature_range=(0,1))
#将**所有**数据归一化，包括特征和标签
vaild_dataset = dataset.drop(columns=['日期','质量等级'])
data_sum = sc.fit_transform(vaild_dataset)

# 将归一化好的数据转化为dataframe(表格)格式，方便后续处理
data_sum = pd.DataFrame(data_sum)

# 划分数据集
x = data_sum.iloc[:, 1:]
y = data_sum[0]

# 划分数据集合
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 加载模型
model = load_model('../Model/air_quality_model.h5')

# 利用训练好的模型预测
predict = model.predict(x_test)
# print(predict,y_test)

# 进行预测值的反归一化
# 将x_test 和 predict放在一起归一化，注意只反归一化predict是不成立的
# 因为归一化会记录列数，归一化和反归一化所需要的参数表格的列数要一致，所以需要将x_test和predict放在一起归一化，再反归一化，怎么归一化过来就要怎么反归一化过去
inv_predict = concatenate((x_test, predict), axis=1)
inv_predict = sc.inverse_transform(inv_predict)
inv_predict_y = inv_predict[:, -1] # 提取反归一化后的预测值
# print(inv_predict_y)

# 将y_test从dataframe（表格）格式转化为 和predict一样的二维数组，以保证前面的正反归一化列数一致
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)
# print(y_test)

# 提取反归一化后的真实值
inv_test_y = concatenate((x_test, y_test), axis=1)
inv_test_y = sc.inverse_transform(inv_test_y)
inv_test_y = inv_test_y[:, -1]

# print(inv_y_test)
# for i in range(len(inv_test_y)):
#     print(inv_predict_y[i],inv_test_y[i],abs(inv_test_y[i] - inv_predict_y[i]))

# 计算rmse 和 mape
rmse = sqrt(mean_squared_error(inv_test_y, inv_predict_y))
mape = np.mean(np.abs((inv_test_y - inv_predict_y) / inv_test_y))

# 打印rmse 和 mape
print('rmse: %.3f' % rmse)
print('mape: %.3f' % mape)

# 绘制预测值和真实值的对比图
plt.plot(inv_predict_y, label='预测值')
plt.plot(inv_test_y, label='真实值')
plt.plot(abs(inv_test_y - inv_predict_y), label='绝对误差')
plt.title('全连接神经网络预测值和真实值对比图')
plt.legend() # 显示图例
plt.show()


