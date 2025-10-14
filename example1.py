from typing import List
import matplotlib.pyplot as plt

#定义数据集
x_data = [1,2,3]

y_data = [2,4,6]

#定义参数
w = 4

# 定义模型
def forward(x) -> int:
    return x * w

#定义损失函数
def cost(x_list: List[int], y_list: List[int]) -> int:
    costValue = 0
    for x,y in zip(x_list, y_list):
        costValue += (forward(x) - y) ** 2
    return costValue / len(x_list) # 均方误差

#定义梯度计算公式
def gradient(x_list: List[int], y_list: List[int]) -> List[float]:
    gradientValue = 0
    for x,y in zip(x_list, y_list):
        gradientValue += 2 * x * (forward(x) - y)
    return gradientValue / len(x_list) # 均方误差的梯度

# 绘制loss曲线
def plot_loss_curve(epoch: int, cost_list: List[float]):
    plt.plot(range(epoch), cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Loss Curve")
    plt.show()

#定义学习率
lr = 0.01

#定义训练次数
epoch = 100

# 定义损失列表
cost_list = []

#训练模型
for i in range(epoch):
    #计算损失并添加到损失列表中
    costValue = cost(x_data, y_data)
    cost_list.append(costValue)
    #计算梯度
    gradientValue = gradient(x_data, y_data)
    #更新参数
    w = w - lr * gradientValue # 梯度下降·更新参数 (w = w - lr * d[loss(w)] / d[w])
    #打印结果
    print(f"epoch: {i + 1}, w: {w}, cost: {costValue}, gradient: {gradientValue}")

# 绘制loss曲线
plot_loss_curve(epoch, cost_list)

#得出训练结论
print(f"经过100轮训练后，w值以及那个训练好了,当x=4是y的预测值为：{forward(4)}")
