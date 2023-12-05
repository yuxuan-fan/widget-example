import torch
from torch import nn
# from torchsummary import summary

class Dropout1(nn.Module):
   def __init__(self):
       super(Dropout1, self).__init__()
       self.fc = nn.Linear(100,20)
 
   def forward(self, input):
       out = self.fc(input)
       out = F.dropout(out, p=0.5, training=self.training)  # 这里必须给traning设置为True
       return out
# 如果设置为F.dropout(out, p=0.5)实际上是没有任何用的, 因为它的training状态一直是默认值False. 由于F.dropout只是相当于引用的一个外部函数, 模型整体的training状态变化也不会引起F.dropout这个函数的training状态发生变化. 所以,在训练模式下out = F.dropout(out) 就是 out = out. 
Net = Dropout1()
Net.train()

#或者直接使用nn.Dropout() (nn.Dropout()实际上是对F.dropout的一个包装, 自动将self.training传入，两者没有本质的差别)
class Dropout2(nn.Module):
  def __init__(self):
      super(Dropout2, self).__init__()
      self.fc = nn.Linear(100,20)
      self.dropout = nn.Dropout(p=0.5)
 
  def forward(self, input):
      out = self.fc(input)
      out = self.dropout(out)
      return out
Net = Dropout2()
Net.train()
