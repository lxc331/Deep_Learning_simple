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
from keras.models import load_model # 导入keras库中的模型加载函数

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
dataset = read_excel('../DataSet/乳腺癌原始数据.xlsx')
# print(dataset)

# 提取特征
x = dataset.iloc[:,1 :-1]  # 提取所有行，除了最后一列
# print(x)

# 提取标签
y = dataset.iloc[:,-1]
# print(y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 将标签转化为独热编码(one-hot格式)，因为在多分类问题中，神经网络无法直接识别类别，容易出现1 + 2 = 3，类别混乱
# 所以需要将标签转化为独热编码，例如：0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_test_one = to_categorical(y_test,2)

# 将数据特征归一化
sc = MinMaxScaler(feature_range=(0,1)) # 归一化到0-1之间
x_test = sc.fit_transform(x_test) # 测试集归一化，

# 加载导入训练好的模型
model = load_model('../Model/breast_cancer_model.h5')

# 模型推理预测结果
predict = model.predict(x_test)
# print(predict)

# 将预测结果转换为类别标签
y_pred = np.argmax(predict, axis=1) # axis=1表示按行取最大值的索引，
# print(y_pred)

# 将0或1转换为良性或恶性
res = []
for x in y_pred:
    if x == 0:
        res.append('良性')
    else:
        res.append('恶性')

print(y_pred,y_test)
report = classification_report(y_test, y_pred, labels=[0,1],target_names=['良性', '恶性'])
print(report)
