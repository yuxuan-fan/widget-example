import torch
from torch import nn
from torchsummary import summary


class ChannelModule(nn.Module):  # 继承nn模块的Module类
    def __init__(self, inputs, ratio=16):  # self必写，inputs接收输入特征张量，ratio是通道衰减因子
        super(ChannelModule, self).__init__()  # 调用父类构造
        _, c, _, _ = inputs.size()  # 获取通道数
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # nn模块的自适应二维最大池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn模块的自适应二维平均池化
        self.share_liner = nn.Sequential(
            nn.Linear(c, c // ratio),
            nn.ReLU(),
            nn.Linear(c // ratio, c)
        )  # 这个共享全连接的3层和SEnet的一模一样，这里借助Sequential这个容器把这3个层整合在一起，方便forward函数去执行，直接调用share_liner(x)相当于直接执行了里面这3层
        self.sigmoid = nn.Sigmoid()  # nn模块的Sigmoid函数

        """
                当你有一个形状为(n, c * 1 * 1)
                的张量时，它在技术上是一个二维形状。它有两个维度：第一个维度的大小是
                n（表示批量大小），第二个维度的大小是
                c * 1 * 1（表示展平后的特征大小）。

                但是，在神经网络中，特别是在处理全连接层时，人们经常将第二个维度称为“展平”或“一维”表示。
                这是因为第二个维度中的每个元素对应于批量中每个样本的唯一特征，使其类似于一维数组。

                因此，虽然张量本身在技术上是二维的，但第二个维度实际上充当了每个批次样本的特征的一维表示。这种展平的表示适合输入到期望一维输入的全连接层中。

        """

    def forward(self, inputs):
        x = self.maxpool(inputs).view(inputs.size(0),-1)
        # 对于输入特征张量，做完最大池化后再重塑形状，view的第一个参数inputs.size(0)表示第一维度，显然就是n；
        # -1表示会自适应的调整剩余的维度，在这里就将原来的(n,c,1,1)调整为了(n,c*1*1)，后面才能送入全连接层（fc层）
        maxout = self.share_liner(x).unsqueeze(2).unsqueeze(3)
        # 做完全连接后，再用unsqueeze解压缩，也就是还原指定维度，这里用了两次，分别还原2维度的h，和3维度的w
        y = self.avgpool(inputs).view(inputs.size(0), -1)
        avgout = self.share_liner(y).unsqueeze(2).unsqueeze(3)  # y走的平均池化路线的代码和x是一样的解释
        return self.sigmoid(maxout + avgout)  # 最后相加两个结果并作归一化


class SpatialModule(nn.Module):
    def __init__(self):
        super(SpatialModule, self).__init__()
        self.maxpool = torch.max
        self.avgpool = torch.mean
        # 和通道机制不一样！这里要进行的是在C这一个维度上求最大和平均，分别用的是torch库里的max方法和mean方法
        self.concat = torch.cat  # torch的cat方法，用于拼接两个张量
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1,
                              padding=3)  # nn模块的二维卷积，其中的参数分别是：输入通道（2），输出通道（1），卷积核大小（7*7），步长（1），灰度填充（3）
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        maxout, _ = self.maxpool(inputs, dim=1,
                                 keepdim=True)  # maxout接收特征点的最大值很好理解，为什么还要一个占位符？因为torch.max不仅返回张量最大值，还会返回索引，索引用不着所以直接忽略，dim=1表示在维度1（也就是nchw的c）上求最大值，keepdim=True表示要保持原来张量的形状
        avgout = self.avgpool(inputs, dim=1, keepdim=True)  # torch.mean则只返回张量的平均值，至于参数的解释和上面是一样的
        outs = self.concat([maxout, avgout],
                           dim=1)  # torch.cat方法，传入一个列表，将列表中的张量在指定维度，这里是维度1（也就是nchw的c）拼接，即n*1*h*w拼接n*1*h*w得到n*2*h*w
        outs = self.conv(outs)  # 卷积压缩上面的n*2*h*w，又得到n*1*h*w
        return self.sigmoid(outs)


class CBAM(nn.Module):
    def __init__(self, inputs):
        super(CBAM, self).__init__()
        self.channel_out = ChannelModule(inputs)  # 获得通道权重
        self.spatial_out = SpatialModule()  # 获得空间权重

    def forward(self, inputs):
        outs = self.channel_out(inputs) * inputs  # 先乘上通道权重
        return self.spatial_out(outs) * outs  # 在乘完通道权重的基础上再乘上空间权重