import numpy as np
import matplotlib.pyplot as plt
 
# 利用 numpy 实现 Sigmoid 函数
def sigmoid(x):
    # x 为输入值
    # y = sigmoid(x)
    y=1/(1+np.exp(-x))
    #dy=y*(1-y)  # 若要实现 Sigmod() 的导数图像，打开此处注释，并返回 dy 值即可。 
    return y
 
# 利用 matplotlib 来进行画图
def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()
 
 
if __name__ == '__main__':
    plot_sigmoid()