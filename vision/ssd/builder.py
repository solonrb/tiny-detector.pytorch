from vision.nn.tinynet import TinyNet,SeperableConv2d
import torch
import torch.nn as nn

def create_tiny_fd(num_class,is_training=False,device='cpu'):
    base_model=TinyNet(2)
    base_model_net=base_model.model
