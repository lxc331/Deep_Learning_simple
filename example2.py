import numpy as np # 导入numpy库
import pandas as pd # 导入pandas库
from sklearn.model_selection import train_test_split # 使用 train_test_split 划分数据集
from sklearn.preprocessing import MinMaxScaler # 特征缩放工具，它通过将每个特征缩放到指定的范围（通常是0到1）来工作，避免量纲(单位)不一致产生的问题
from sklearn.linear_model import LogisticRegression # 导入线性回归模型
from sklearn.metrics import classification_report # 分类报告，用于评估分类模型的性能

# 读取数据
dataset = pd.read_excel('./DataSet/乳腺癌原始数据.xlsx')
# print(dataset)

# 提取数据特征x
x = dataset.iloc[:,1:-1] # 提取所有行，除了最后一列和第一列(第一列是id)
# print(x)

# 提取数据标签y
y = dataset.iloc[:,-1]
# print(y)

# 划分数据集
# 划分数据集，测试集占20%，随机种子42，random state 42 是一种常见的实践，用于控制随机性，确保每次运行结果一致
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print(x_train)

# 特征缩放(归一化)，避免量纲(单位)不一致产生的问题
# 量纲不一致，例如：长度、重量、时间等
# 归一化，将数据映射到[0,1]区间
# 归一化的公式：x = (x - min) / (max - min)
# 归一化的作用：1. 避免量纲不一致产生的问题 2. 提高模型的收敛速度 3. 提高模型的精度
# 归一化的注意事项：1. 归一化只适用于数值型特征 2. 归一化不会改变特征的分布 3. 归一化不会改变特征的关系 4. 归一化只适用于训练集，测试集要使用训练集的参数进行归一化
scaler = MinMaxScaler(feature_range=(0,1))

# 归一化训练集
x_train = scaler.fit_transform(x_train)
# 归一化测试集
x_test = scaler.fit_transform(x_test)
# print(x_train)

# 到这里数据处理完成，开始模型训练
lr = LogisticRegression() #实例化模型
lr.fit(x_train, y_train) # 训练模型

# 打印模型参数
# print('模型系数w:',lr.coef_) # 有多少个属性就有多少w
# print('模型截距b:',lr.intercept_) # b只有一个

# 测试预测结果
prediction = lr.predict(x_test)
# print('预测结果:',prediction)

# 打印逻辑回归概率，逻辑回归的计算结果是一个概率值，用概率表示结果，小于0.5为类别0同时为良性肿瘤，大于0.5为类别1同时为恶性肿瘤，(逻辑回归的计算结果是一个概率值)
pre_result_probe = lr.predict_proba(x_test)
# print('逻辑回归概率:',pre_result_probe) # 打印[良性肿瘤概率（0）,恶性肿瘤概率（1）]
# print('打印恶性肿瘤概率:\n',pre_result_probe[:,1])
# print('打印良性肿瘤概率:\n',pre_result_probe[:,0])

# 重新设置阈值(默认阈值一般为0.5)
threshold = 0.3 # 0.3为阈值，大于0.3为恶性肿瘤，小于0.为良性肿瘤(为了严谨)

# 设置保存结果的列表
result = []
result_name = []

# 遍历概率列表，根据阈值判断肿瘤类型
for i in pre_result_probe[:,1]:
    if i > threshold:
        result.append(1)
        result_name.append('恶性肿瘤')
    else:
        result.append(0)
        result_name.append('良性肿瘤')

# 打印阈值调整后的结果
# print('阈值调整后的结果:',result)
# print('阈值调整后的结果名称:',result_name)

# 评估模型的精确度，召回率，F1值
report = classification_report(y_test, result, labels = [0,1],target_names= ['良性肿瘤','恶性肿瘤']) # 参数1：真实标签，参数2：预测标签
print('模型评估结果:\n',report)
