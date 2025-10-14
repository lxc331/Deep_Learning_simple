import numpy as np # 用于处理数组
import pathlib # 用于处理文件路径
import matplotlib.pyplot as plt # 用于绘制图表
import pandas as pd # 用于处理数据框
import keras # 用于构建和训练神经网络模型

# 分别从keras的layers(分层包)中导入全连接层，平展层(将2维图片展开为1维神经元)，卷积层，池化层
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.python.ops.nn_impl import sufficient_statistics

# 解决中文显示问题，以及符号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取训练集数据
data_train = pathlib.Path('DataSet/insect_identification/train')
#读取验证集数据
data_val = pathlib.Path('DataSet/insect_identification/val')

# 把数据类别放置在列表数据中
CLASS_NAME = np.array(['ants', 'bees'])

# 设置图片大小和批次数
BATCH_SIZE = 100 # 每个批次的样本数量
IMG_HEIGHT = 64 # 图片的高度
IMG_WIDTH = 64 # 图片的宽度

# 对数据进行归一化处理，统一量纲加快收敛速度
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
# 训练集生成器
# 从训练集数据中读取数据, 并且统一图片格式，即归一化处理
train_data_generator = image_generator.flow_from_directory(
    directory = str(data_train), # 训练集数据的路径
    target_size = (IMG_HEIGHT, IMG_WIDTH), # 图片的目标大小
    batch_size = BATCH_SIZE, # 每个批次的样本数量
    shuffle = True, # 是否随机打乱数据
    classes = list(CLASS_NAME) # 数据的类别
)
# 验证集生成器
# 从验证集数据中读取数据, 并且统一图片格式，即归一化处理
val_data_generator = image_generator.flow_from_directory(
    directory = str(data_val),
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    shuffle = True,
    classes = list(CLASS_NAME)
)

# 利用keras的Sequential模型构建卷积神经网络模型
model = keras.Sequential()
# 卷积层1
# filters: 卷积核的数量
# kernel_size: 卷积核的大小
# activation: 激活函数
# input_shape: 输入图片的大小，(图片的高度, 图片的宽度, 图片的通道数(这里是输入三通道彩色图))
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
# 池化层1
# pool_size: 池化窗口(池化核)的大小
# strides: 池化窗口的步幅
model.add(MaxPool2D(pool_size=(2,2), strides=2))
# 卷积层2
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
# 池化层2
model.add(MaxPool2D(pool_size=(2,2), strides=2))
# 卷积层3
model.add(Conv2D(filters=120, kernel_size=(5,5), activation='relu'))
# 平展层, 将2维图片展开为1维神经元
model.add(Flatten())
# 全连接层1
model.add(Dense(84, activation='relu'))
# 输出层
model.add(Dense(2, activation='softmax'))

# 编译模型
# optimizer: 优化器，用于更新模型的权重(adam优化器)
# loss: 损失函数，用于计算模型的误差(交叉熵损失函数)
# metrics: 评估指标，用于评估模型的性能
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
# epochs: 训练的轮数
# validation_data: 验证集数据，用于评估模型的性能
history = model.fit(train_data_generator, epochs=50, validation_data=val_data_generator)

# 保存训练好的模型
model.save('./Model/insect_identification.h5')

# 绘制loss值
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('CNN神经网络模型损失值变化图')
plt.legend()
plt.show()

# 绘制accuracy值
plt.plot(history.history['accuracy'],label='train_accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('CNN神经网络模型准确率变化图')
plt.legend()
plt.show()
