import pandas as pd
import torch
import matplotlib.pyplot as plt


# linear model : y = wx + b

# 1. read the data
data = pd.read_csv('./line_fit_data.csv').values
X = torch.tensor(data[:,0],dtype=torch.float32)
y = torch.tensor(data[:,1],dtype=torch.float32)

W = torch.tensor(-10.0, requires_grad=True)  # w设定任意初始值，待更新，设定需要梯度为真
b = torch.tensor(7.0, requires_grad=True)    # 同上
learning_rate = 0.35

# 2. construct a linear model
def linear_model(W, X, b):
    return W * X + b

def loss_fun(y_pre, y_true):                  # 自定义的损失函数
    return ((y_pre-y_true)**2).mean()

# 绘制数据分布和初始模型
plt.figure(figsize=(10,5))      # 设置画布大小
plt.axis([-0.01, 1, -3, 10])   # 指定坐标轴的取值范围
plt.scatter(X, y, color='cyan')   # 绘制样本实际分布图
plt.plot(X, linear_model(W, X, b).data, color='magenta')     # 绘制模型预测结果分布
plt.show()
plt.legend(['y_true', 'model_pred'])                    # 设置图例
