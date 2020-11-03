import torch
import torch.nn as nn


 class SSD (nn.Module):
     def __init__(self,base_model,classification_header,regression_header,source_layers_index,extra_layer,config=None,training=False,device=None):
         '''
         classification_header/regression_header/base_model:  torch.nn.MduleList
         sourec_layers_index: List[int],the index of source_layer to predictoin location and score
         config: config for ssd arch params
         '''
        super(SSD,self).__init__()
        self.base_model=base_model
        self.classification_header=classification_header
        self.regression_header=regression_header
        self.source_layers_index=source_layers_index
        self.config=config
        self.training=training
        self.extra_layer=extra_layer
        if device:
            self.device=device
        else:
            self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.priors=self.config.priors
        # self.confidences=[]
        # self.locations=[]
    def forward(self,x):
        confidences=[]
        locations=[]
        end_layer_index=0
        start_layer_index=0
        header_index=0
        for index in self.source_layers_index:
            for layer in self.base_model[start_layer_index:end_layer_index]:
                x=layer(x)
            location=self.regression_header[header_index](x)
            location=location.permute(0,2,3,1).contiguous()
            confidence=self.classification_header[header_index](x)
            confidence=confidence.permute(0,2,3,1).contiguous()         # n c h w---->n h w c
            confidences.append(confidence)
            locations.append(location)
            start_layer_index=end_layer_index
            header_index=header_index+1
        for layer in self.base_model[end_layer_index:]:
            x=layer(x)
        for ex_layer in self.extra_layer:
            x=extra_layer(x)
            confidence=self.classification_header[header_index](x)
            location=self.regression_header[index](x)
            confidence=confidence.permute(0,2,3,1).contiguous()
            location-location.permute(0,2,3,1).contiguous()
            


