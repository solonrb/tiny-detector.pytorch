import torch
import torch.nn as nn


class SSD(nn.Module):
    def __init__(self,num_class,base_model,classification_header,regression_header,source_layers_index,extra_layer,fpn_module=None,config=None,training=False,device=None):
        super(SSD,self).__init__()
        self.num_class=num_class
        self.base_model=base_model
        self.classification_header=classification_header
        self.regression_header=regression_header
        self.source_layers_index=source_layers_index
        self.config=config
        self.training=training
        self.extra_layer = extra_layer
        self.fpn=fpn_module
        if device:
         self.device=device
        else:
         self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self,x):
        confidences=[]
        locations=[]
        end_layer_index=0
        start_layer_index=0
        header_index=0
        for index in self.source_layers_index:
            end_layer_index=index
            for layer in self.base_model[start_layer_index:end_layer_index]:
                x=layer(x)
            location=self.regression_header[header_index](x).permute(0,2,3,1).contiguous()
            location=location.view(location.size(0),-1,4)
            confidence=self.classification_header[header_index](x).permute(0,2,3,1).contiguous()
            confidence=confidence.view(confidence.size(0),-1,self.num_class)             # make sure confidence shape is n num_priors,4
            confidences.append(confidence)
            locations.append(location)
            start_layer_index=end_layer_index
            header_index+=1
        for layer in self.base_model[end_layer_index:]:
            x=layer(x)
        for ex_layer in self.extra_layer:
            x=ex_layer(x)
            location=self.regression_header[header_index](x).permute(0,2,3,1).contiguous()
            confidence=self.classification_header[header_index](x).permute(0,2,3,1).contiguous()
            location = location.view(location.size(0), -1, 4)
            confidence = confidence.view(confidence.size(0), -1, self.num_class)
            locations.append(location)
            confidences.append(confidence)
        if not self.training:
            confidences=nn.functional.softmax(confidences,dim=-1)
            return (confidences,locations)
        else:
            return (confidences,locations)
    def init_param(self):
        self.base_model.apply(self.init_func)
        self.regression_header.apply(self.init_func)
        self.classification_header.apply(self.init_func)
        self.extra_layer.apply(self.init_func)
    def init_func(self,m):
        if isinstance(m,nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)




            


