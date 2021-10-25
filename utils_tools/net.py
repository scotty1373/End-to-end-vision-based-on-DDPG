# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import itertools


class Common(nn.Module):
    def __init__(self):
        super(Common, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(4, 4),
                               padding=(1, 1))
        self.actv2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv4 = nn.LeakyReLU(inplace=True)

        self.inputDense1 = nn.Linear(4, 64)

    def forward(self, x1, x2):
        feature = self.conv1(x1)
        feature = self.actv1(feature)
        feature = self.conv2(feature)
        feature = self.actv2(feature)
        feature = self.conv3(feature)
        feature = self.actv3(feature)
        feature = self.conv4(feature)
        feature = self.actv4(feature)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)

        out = self.inputDense1(x2)

        output = torch.cat([feature, out], dim=1)
        return output


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # self.common = Common()
        self.Dense1 = nn.Linear(3264, 400)
        self.actv1 = nn.LeakyReLU(inplace=True)
        self.Dense2 = nn.Linear(400, 300)
        self.actv2 = nn.LeakyReLU(inplace=True)
        self.Dense3 = nn.Linear(300, 1)
        torch.nn.init.uniform_(self.Dense3.weight, a=-3e-3, b=3e-3)
        self.actv3 = nn.Tanh()

    def forward(self, common):
        action = self.Dense1(common)
        action = self.actv1(action)
        action = self.Dense2(action)
        action = self.actv2(action)
        action = self.Dense3(action)
        action = self.actv3(action)
        return action


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.inputDense1 = nn.Linear(1, 16)
        self.inputactv = nn.LeakyReLU(inplace=True)
        self.inputDense2 = nn.Linear(16, 32)
        self.Dense1 = nn.Linear(3264 + 32, 400)
        self.actv3 = nn.LeakyReLU(inplace=True)
        self.Dense2 = nn.Linear(400 + 32, 300)
        self.actv4 = nn.LeakyReLU(inplace=True)
        self.Dense3 = nn.Linear(300, 1)
        torch.nn.init.uniform_(self.Dense3.weight, a=-3e-4, b=3e-4)


    def forward(self, common, action):
        input_action = self.inputDense1(action)
        input_action = self.inputactv(input_action)
        input_action = self.inputDense2(input_action)
        concat_layer1 = torch.cat([common, input_action], dim=1)
        critic = self.Dense1(concat_layer1)
        critic = self.actv3(critic)
        critic_concat = torch.cat([critic, input_action], dim=1)
        critic_concat = self.Dense2(critic_concat)
        critic_concat = self.actv4(critic_concat)
        critic_concat = self.Dense3(critic_concat)
        return critic_concat


if __name__ == '__main__':
    common_net = Common()
    actor_net = Actor()
    critic_net = Critic()
    loss = torch.nn.MSELoss()
    opt_common = torch.optim.SGD(common_net.parameters(), 0.001)
    # opt_actor = torch.optim.SGD(itertools.chain(common_net.parameters(), actor_net.parameters()), 0.001)
    opt_actor = torch.optim.SGD(actor_net.parameters(), 0.001)
    opt_critic = torch.optim.SGD(itertools.chain(common_net.parameters(), actor_net.parameters()), 0.001)
    print(common_net)
    print(actor_net)
    print(critic_net)

    x = torch.randn((10, 4, 80, 80))
    y = torch.randn((10, 4))

    out1 = actor_net(common_net(x, y))
    out2 = critic_net(common_net(x, y), out1)

    tgt = torch.rand(10, 1)

    loss_scale = loss(out1, tgt)
    opt_actor.zero_grad()
    loss_scale.backward()
    opt_actor.step()



