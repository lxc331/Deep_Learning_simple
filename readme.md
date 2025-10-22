## 该项目对应的是b站up主 '炮哥带你学' 的深度学习课程

### 视频位置: 
[点击该网址进入视频](https://www.bilibili.com/video/BV1eP411w7Re?spm_id_from=333.788.videopod.episodes&vd_source=c8e3d805e4bab5aff6e3790eb7863c38)

---
### python解释器
#### 该项目使用的python解释器版本为python3.8

---

### 所需python环境
- python 3.8 

### 需要导入的包
#### 该项目需要导入的包有：
- tensorflow 2.4.0 
- keras 2.4.3 
- numpy 1.19.5 
- pandas 1.3.5 
- matplotlib 3.4.2 
- sklearn 0.0
- opencv-python 4.3.0.38 

导包指令，在conda环境下的Anconda prompt使用命令行导入：类似
```
# 可以自己换源，我这里用的是清华源，也可以用豆瓣源
pip install opencv-python==4.3.0.38 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

---
### 安装命令
#### 分别对应安装命令为：
- pip install opencv-python==4.3.0.38 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.douban.com

---

### example1.py
#### 线性回归实战(回归任务),任务是预测某一学习时间下取得什么成绩，数据集自己构造

### example2.py:
#### 逻辑回归实战(分类任务)，任务是根据肿瘤的半径、纹理、周长等特征，判断肿瘤是良性还是恶性，数据集使用[乳腺癌原始数据.xlsx](DataSet/%E4%B9%B3%E8%85%BA%E7%99%8C%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE.xlsx)

### example3.py:
#### 全连接神经网络(分类任务)实战，任务是根据肿瘤的半径、纹理、周长等特征，判断肿瘤是良性还是恶性，数据集使用[乳腺癌原始数据.xlsx](DataSet/%E4%B9%B3%E8%85%BA%E7%99%8C%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE.xlsx)

### example4.py:
#### 全连接神经网络(回归任务)实战，任务是根据空气质量数据，预测下一个小时的空气质量，数据集使用[AirQuality_ShiJiaZhuang.csv](DataSet/AirQuality_ShiJiaZhuang.csv)

### example5.py:
#### 卷积神经网络(分类任务)实战，任务是根据图片判断是6种不同的钢板缺陷，数据集使用[钢板缺陷](DataSet/surfaceDefect)
#### 注：这里用的是LeNet模型，详细图如下：
![LeNet-5卷积神经网络模型图示.png](ModelGraphic/LeNet%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B%E5%9B%BE%E7%A4%BA.png)![LeNet模型](./ModelGraphic/LeNet.png)

### example6.py:
#### LSTM模型实战，任务是预测黄金价格，数据集使用[黄金价格](DataSet/gold.csv)
