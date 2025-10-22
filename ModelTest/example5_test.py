import numpy as np # 用于处理数组
import tensorflow as tf # 用于构建和训练神经网络模型
from keras.models import load_model # 用于加载训练好的模型
import cv2 # 用于处理图片
import matplotlib.pyplot as plt # 用于绘制图表


# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 把数据类别放置在列表数据中
CLASS_NAME = np.array(['crazing', 'inclusion','patches','pitted_surface','rolled-in_scale','scratches'])

# 设置图片大小
IMG_HEIGHT = 32 # 图片的高度
IMG_WIDTH = 32 # 图片的宽度

# 加载模型
model = load_model('../Model/surfaceDefect.h5')

# 读取图片与预处理
path = '../DataSet/surfaceDefect/test/'
path += input('请输入测试图片的文件名：')
src = cv2.imread(path) # 读取图片
src = cv2.resize(src, (IMG_WIDTH, IMG_HEIGHT)) # 调整图片大小
src = src.astype('int32') # 转换为int32类型
src = src / 255  # 归一化处理
# 扩展维度
test_image = tf.expand_dims(src, 0) # 扩展维度，增加一个维度，用于表示批量大小
# print(test_image.shape) # 打印图片的形状数据，打印结果(1, 64, 64, 3)表示批次为1，大小64*64，三通道

# 预测
prediction = model.predict(test_image) # 预测结果是每个类别的概率，打印结果如[[0.99795425 0.00204571]]，再将概率转化为类别
score = prediction[0] # 二维数组，取第一行，如[0.99795425 0.00204571]

# 打印预测结果
print('预测结果为：{}，概率为：{} '.format(CLASS_NAME[np.argmax(score)], max(score))) # argmax(score)表示取列表的最大值的索引
