# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np


class Data_dim_reduce(nn.Module):
    def __init__(self):
        super(Data_dim_reduce, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8),
                               stride=(4, 4))
        self.activation1 = nn.ReLU(inplace=True)
        # 在keras中，通过卷积计算的输出图像如果不为整数，则只取整数部分，相当于留有未卷积的边
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4),
                               stride=(4, 4), padding=(1, 1))
        self.activation2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.activation3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.Dense1 = nn.Linear(3200, 128)
        self.activationDense = nn.ReLU(inplace=True)
        self.Dense2 = nn.Linear(128, 21)

        self.inputDense1 = nn.Linear(4, 16)
        self.inputDense2 = nn.Linear(16, 21)

        self.catDense = nn.Linear(42, 21)

        # 简洁forward写法， 将图形特征提取打成块，可避免forward过于冗长
        # self.figure_feature = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=16,
        #                                               kernel_size=(8, 8), stride=(4, 4)),
        #                                     nn.ReLU(True),
        #                                     nn.Conv2d(in_channels=16, out_channels=32,
        #                                               kernel_size=(4, 4), stride=(4, 4), padding=(1, 1)),
        #                                     nn.ReLU(True),
        #                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
        #                                               stride=(1, 1), padding=(1, 1)),
        #                                     nn.ReLU(True),
        #                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
        #                                               stride=(1, 1), padding=(1, 1)))
        print('Model we built ready!!!')

    def forward(self, x1, x2):
        feature = self.conv1(x1)
        feature = self.activation1(feature)
        feature = self.conv2(feature)
        feature = self.activation2(feature)
        feature = self.conv3(feature)
        feature = self.activation3(feature)
        feature = self.conv4(feature)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)
        feature = self.Dense1(feature)
        feature = self.activationDense(feature)
        feature = self.Dense2(feature)

        velocity = self.inputDense1(x2)
        velocity = self.inputDense2(velocity)

        concatenate = torch.cat([feature, velocity], dim=1)
        logits = self.catDense(concatenate)
        return logits


if __name__ == "__main__":
    net = Data_dim_reduce()
    x = np.random.normal(size=(10, 4, 80, 80))
    y = np.random.normal(size=(10, 4))
    net(x, y)

    epoch = 233
    # 参数保存和读取
    torch.save({'epochs': epoch, 'model': net.state_dict(), 'optimizer': optim.state_dict()}, '3in1.pt')

    checkpoints = torch.load('3in1.pt')
    model = Data_dim_reduce()
    optim = torch.optim.Adam(model.parameters())
    model.load_state_dict(checkpoints['model'])
    optim.load_state_dict(checkpoints['optimizer'])
    epochs = checkpoints['epochs']