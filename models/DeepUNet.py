import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

class down_block(nn.Module):
    def __init__(self, input_c, output_c):
        super(down_block, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(input_c,output_c,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(output_c, eps=1e-5, momentum=0.99),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(input_c)
        self.conv = nn.Conv2d(output_c, input_c, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.max_pool(x) #64x320x320
        tmp = self.conv_bn_relu(x) #64x320x320
        bn = self.bn(self.conv(tmp))
        bn = bn + x
        act = self.relu(bn)
        return bn, act

class up_block(nn.Module):
    def __init__(self, input_c, output_c):
        super(up_block, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(input_c,input_c,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(input_c, eps=1e-5, momentum=0.99),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(output_c)
        self.conv = nn.Conv2d(input_c, output_c, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x1, x2):
        x = self.upsample(x1) #32x20x20
        tmp = torch.cat([x, x2], dim=1) #64x20x20
        tmp = self.conv_bn_relu(tmp) #64x20x20
        bn = self.bn(self.conv(tmp)) #32x20x20
        bn = bn + x
        act = self.relu(bn)
        return bn

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class deepunet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(deepunet, self).__init__()
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.99),
            nn.ReLU()
        )
        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.99),
            nn.ReLU()
        )
        self.bn1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32,momentum=0.99)
        )
        self.relu = nn.ReLU()
        self.down_block = down_block(32, 64)
        self.up_block = up_block(64,32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x = self.conv_bn_relu1(x) #32x640x640
        net = self.conv_bn_relu2(x) #64x640x640
        bn1 = self.bn1(net) #32x640x640
        act1 = self.relu(bn1) #32x640x640

        bn2, act2 = self.down_block(act1) #32x320x320, 32x320x320
        bn3, act3 = self.down_block(act2) #32x160x160
        bn4, act4 = self.down_block(act3)
        bn5, act5 = self.down_block(act4)
        bn6, act6 = self.down_block(act5)
        bn7, act7 = self.down_block(act6) #32x10x10

        tmp = self.up_block(act7, bn6)
        tmp = self.up_block(tmp, bn5)
        tmp = self.up_block(tmp, bn4)
        tmp = self.up_block(tmp, bn3)
        tmp = self.up_block(tmp, bn2)
        tmp = self.up_block(tmp, bn1) 
        tmp = self.out(tmp)
        return tmp
        


        
# if __name__ == '__main__':
#     x1 = Variable(torch.randn(3, 3, 320, 320))
#     # x2 = Variable(torch.randn(3, 32, 20, 20))
#     # tmp = up_block(64, 32)
#     # y = tmp(x1, x2)
#     # tmp = down_block(32, 64)
#     # bn, act = tmp(x)
#     net = deepunet(3,3)
#     y = net(x1)
#     print(y)
