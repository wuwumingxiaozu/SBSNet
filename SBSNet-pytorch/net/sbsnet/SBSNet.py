from net.sbsnet.resnet import *
from net.sbsnet.sknet import *
from net.sbsnet.enhanced_feature_net import *


class SBSNet(nn.Module):
    def __init__(self, num_classes):
        super(SBSNet, self).__init__()
        operation_down = [[3, 64], 'D', [64, 128], 'D', [128, 256], 'D', [256, 512], 'D', [512, 1024]]
        self.res = RESNet(operation_down)
        self.sk = SKNet(operation_down, 1, 2, 16, 32)

        operations_up = ['U', [1024, 512], 'U', [512, 256], 'U', [256, 128], 'U', [128, 64]]
        self.up = UPNet(operations_up)

        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        resR = self.res(x)
        skR = self.sk(x)
        up_input = list(map(lambda x, y: x + y, resR, skR))
        up = self.up(up_input)
        out = self.out(up)

        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = SBSNet(1)
    print(net(x).shape)
