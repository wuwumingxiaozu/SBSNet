import torch
from torch import nn
from torch.nn import functional as F
from net.sbsnet.resnet import *
from net.sbsnet.sknet import *

'''上采样'''


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU()
        )

    def forward(self, x, feature_map):
        out = self.up(x)
        return torch.cat((out, feature_map), dim=1)


# class UpSample(nn.Module):
#     def __init__(self, channel):
#         super(UpSample, self).__init__()
#         self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
#
#     def forward(self, x, feature_map):
#         up = F.interpolate(x, scale_factor=2, mode='nearest')
#         out = self.layer(up)
#         return torch.cat((out, feature_map), dim=1)


class UPNet(nn.Module):
    def __init__(self, operations):
        super(UPNet, self).__init__()
        op = operations[1][0]
        self.convs = nn.ModuleList()
        for operation in operations:
            if operation == 'U':
                self.convs.append(UpSample(op))
            else:
                self.convs.append(Conv_RES(operation[0], operation[1]))
                op = operation[1]

    def forward(self, list_tensorlist):
        R6 = self.convs[1](self.convs[0](list_tensorlist[0], list_tensorlist[1]))
        R7 = self.convs[3](self.convs[2](R6, list_tensorlist[2]))
        R8 = self.convs[5](self.convs[4](R7, list_tensorlist[3]))
        R9 = self.convs[7](self.convs[6](R8, list_tensorlist[4]))

        return R9


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    print(x)
    operations = [[3, 64], 'D', [64, 128], 'D', [128, 256], 'D', [256, 512], 'D', [512, 1024]]
    sknet = SKNet(operations, 1, 2, 16, 32)
    resnet = RESNet(operations)
    resR = resnet(x)
    skR = sknet(x)
    upinput = list(map(lambda x, y: x + y, resR, skR))
    operations = ['U', [1024, 512], 'U', [512, 256], 'U', [256, 128], 'U', [128, 64]]
    upnet = UPNet(operations)
    out = upnet(upinput)
    print(out)
    print(out.shape)
