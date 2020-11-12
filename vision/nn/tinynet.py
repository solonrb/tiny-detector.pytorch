import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyNet(nn.Module):
    def __init__(self,num_class):
        super(TinyNet,self).__init__()
        self.num_class=num_class
        self.base_channel=8*2
        def conv_dw(in_channels,out_channels,stride):
            return nn.Sequential(
                nn.Conv2d(in_channels,in_channels,3,stride,padding=1,groups=in_channels,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels,out_channels,1,1,padding=0,bias=False)
            )
        def conv_bn(in_channels,out_channels,stride):
            return nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,stride,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) 
            )
        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1)
        )
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self,x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



def SeperableConv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding=padding,groups=in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,1,1,padding=0),
    )
