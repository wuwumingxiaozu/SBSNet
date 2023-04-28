import torch
from torch import nn
from net.sbsnet.downsample import *

'''
    SKNet
    :param  in_channels:    输入通道数
    :param  out_channels:   输出通道数
    :param  stride:         步长，默认为1
    :param  M:              分支数
    :param  r:              特征Z的长度，计算长度
    :param  L:              论文中规定特征Z的下界，默认为32
'''


class Conv_SK(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(Conv_SK, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True)
        )
        # 计算向量Z的长度d
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        # 根据分支数量，添加不同核的卷积操作
        self.convs = nn.ModuleList()
        for i in range(M):
            # 为提高效率，原论文中，扩张卷积5*5为（3*3，dilation=2）来代替，且论文中建议分组卷积groups=32
            # dilation=1为普通卷积，dilation=2为相隔1的空洞卷积
            self.convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        # 自适应 pool 带指定维度 -GAP
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 降维
        self.fc = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 升维
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, out_channels, kernel_size=1, stride=1, bias=False)
            )

        # 指定dim=1 令两个FCS对应位置进行softmax，保证对应位置a+b+..=1
        self.softmax = nn.Softmax(dim=1)

        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 获得批次
        batch_size = x.size(0)
        x = self.conv(x)

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        # 将原始输入
        feats = feats.view(batch_size, self.M, self.out_channels, feats.shape[2], feats.shape[3])
        # the part of fusion
        # 逐元素相加生成 混合特征U
        feats_U = torch.sum(feats, dim=1)
        # 全局平均池化，将U转化为1*1*C的
        feats_S = self.global_pool(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.out_channels, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class SKNet(nn.Module):
    def __init__(self, operations, stride=1, M=2, r=16, L=32):
        super(SKNet, self).__init__()
        op = operations[0][1]
        self.convs = nn.ModuleList()
        for operation in operations:
            if operation == 'D':
                self.convs.append(DownSample(op))
            else:
                self.convs.append(Conv_SK(operation[0], operation[1], stride, M, r, L))
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
    net = SKNet(operations, 1, 2, 16, 32)
    print(net(x)[2].shape)
