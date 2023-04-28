import torch
from torch import nn
from net.sbsnet.downsample import *

'''
    RESNet
    :param  in_channel:     输入通道数
    :param  out_channel:    输出通道数
'''


class Conv_RES(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_RES, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.layer1 = nn.Sequential(
            # 卷积操作
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            # 基于通道的批量归一化
            nn.BatchNorm2d(self.out_channel),
            # 随机刨除掉30%的数据防止过拟合
            nn.Dropout2d(0.3),
            # 激活函数
            nn.LeakyReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.Dropout2d(0.3)
        )
        if self.in_channel != self.out_channel:
            self.layer2 = nn.Sequential(
                # 卷积操作
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1),
                # 基于通道的批量归一化
                nn.BatchNorm2d(self.out_channel),
                nn.Dropout2d(0.3)
            )

        self.relu = nn.LeakyReLU()

    def forward(self, image):
        if self.in_channel == self.out_channel:
            return self.relu(self.layer1(image) + image)
        else:
            return self.relu(self.layer1(image) + self.layer2(image))


class RESNet(nn.Module):
    def __init__(self, operations):
        super(RESNet, self).__init__()
        op = operations[0][1]
        self.convs = nn.ModuleList()
        for operation in operations:
            if operation == 'D':
                self.convs.append(DownSample(op))
            else:
                self.convs.append(Conv_RES(operation[0], operation[1]))
                op = operation[1]

    def forward(self, x):
        R1 = self.convs[0](x)
        R2 = self.convs[2](self.convs[1](R1))
        R3 = self.convs[4](self.convs[3](R2))
        R4 = self.convs[6](self.convs[5](R3))
        R5 = self.convs[8](self.convs[7](R4))

        return R5, R4, R3, R2, R1


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    print(x)
    operations = [[3, 64], 'D', [64, 128], 'D', [128, 256], 'D', [256, 512], 'D', [512, 1024]]
    net = RESNet(operations)
    print(net(x)[4].shape)
