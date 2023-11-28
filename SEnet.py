import torch
from torch import nn
from torchsummary import summary


class SEAttention(nn.Module):
    def __init__(self, inputs, ratio=4):  # self必须写，inputs接收输入张量，ratio是通道衰减因子
        super(SEAttention, self).__init__()  # super关键字调用父类(即nn.Moudule类)的构造方法
        _, c, _, _ = inputs.size()  # 获取张量的形状(即NCHW)，该模块只关注参数C，其余用占位符忽略
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn模块的自适应二维平均池化，参数1等同于全局平均池化
        self.linear1 = nn.Linear(c, c // ratio,
                                 bias=False)  # nn模块的全连接，这里输入c，输出c//ratio，bias是偏置参数，网络层是否有偏置，默认存在，若bias=False，则该网络层无偏置，图层不会学习附加偏差
        self.relu = nn.ReLU(inplace=True)  # nn模块的ReLU激活函数，inplace=True表示要用引用传递（即地址传递），估计可以减少张量的内存占用（因为值传递要拷贝一份）
        self.linear2 = nn.Linear(c // ratio, c, bias=False)  # 同全连接1，但输入输出相反
        self.sigmoid = nn.Sigmoid()  # nn模块的Sigmoid函数

    # 代码逐行解释：
    def forward(self, inputs):  # self必须写，inputs接收输入特征张量
        n, c, _, _ = inputs.size()  # 获取张量形状（即NCHW），HW被忽略
        x = self.avgpool(inputs).view(n,
                                      c)  # nchw，池化加view方法重塑（reshape）张量形状，因为全连接层之间的张量必须是二维的（一个输入维度一个输出维度），view的参数是(n,c)表示只保留这两个维度
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)  # 上面这四行直接调用初始化好的网络层即可
        x = x.view(n, c, 1, 1)  # reshape还原维度，因为要和原输入特征相乘，不重塑形状不同无法相乘
        return inputs * x  # 和原输入特征层相乘


# 这边是测试代码，用summary类总结网络模型层
inputs = torch.randn(32, 512, 26, 26)  # NCHW
my_model = SEAttention(inputs)
outputs = my_model(inputs)
summary(my_model.cuda(), input_size=(512, 26, 26))