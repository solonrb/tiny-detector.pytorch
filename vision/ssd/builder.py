import torch
import torch.nn as nn
from vision.nn.tinynet import SeperableConv2d,TinyNet
from vision.ssd.ssd import SSD

'''
create handle for full ssd arch
'''
def create_net_hd(num_class,config=None,training=True,device='cuda'):
    net=TinyNet(num_class)
    base_model=net.model
    # builder the prediction header
    classification_headers=nn.ModuleList([
        SeperableConv2d(in_channels=net.base_channel*4,out_channels=3*num_class)
    ])

