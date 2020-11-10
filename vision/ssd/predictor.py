import torch
from vision.uilts import box_utils


class Predictor(object):
    def __init__(self,net,device='cpu'):
        self.net=net
        self.device=device
        if self.device:
            net.to(torch.device(self.device))
        net.eval()
    def __call__(self,data):
        data=data.to(torch.device(self.device))
        with torch.no_grad:
            locations,probs=self.net(data)
