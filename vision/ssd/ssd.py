import torch
import torch.nn as nn
from vision.uilts import box_utils

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
    def init_base_from_prtrain(self,model_path):
        self.base_model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage),strict=True)
        self.extra_layer.apply(self.init_func)
        self.regression_header.apply(self.init_func)
        self.classification_header.apply(self.init_func)

class Anchor(object):
    def __init__(self,priors,center_variance,size_variance,iou_threshold):
        self.priors=priors  #  (x,y,w,h)
        self.corner_from_priors=self.center_from_to_corner(self.priors)
        self.center_variance=center_variance
        self.size_variance=size_variance
        self.iou_threshold=iou_threshold
    def center_from_to_corner(self,center_from_priors)->torch.Tensor:
        return torch.cat([center_from_priors[...,:2]-center_from_priors[...,2:]/2,center_from_priors[...,:2]+center_from_priors[...,2:]/2],dim=center_from_priors.dim()-1)
    def corner_from_to_center(self,corner_from_priors)->torch.Tensor:
        return torch.cat([(corner_from_priors[...,:2]+corner_from_priors[...,2:])/2,corner_from_priors[...,2:]-corner_from_priors[...,:2]],dim=corner_from_priors.dim()-1)
    def encode(self,gt_boxes,gt_labels)->tuple(torch.Tensor,torch.Tensor):
        '''
        function decode gt boxes into train format
        Args:
            gt_boxes: (num_targets,4) -->(x1,y1,x2,y2)
            gt_labels: (num_targets)
            priors:(num_priors,4)
        Returns:
            tuple(torch.Tensor,torch.Tensor)
            locations:shape of (num_priors,4)
            labels:shape of (num_priors)
        '''
        # size of iou score is [num_priors,num_targets]
        iou=box_utils.iou(gt_boxes.unsqueeze(0),self.corner_from_priors.unsqueeze(1))     # shape of iou is [num_priors,num_targets]
        # size of best_target_per_prior_index -->[num_prior]
        best_target_per_prior,best_target_per_prior_index=iou.max(1)
        # best_prior_per_target_index  --->[num_target]
        # 为每个target box筛选最匹配的先验框
        best_prior_per_target,best_prior_per_target_index=iou.max(0)
        # assign best target to a prior
        for index,prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index]=index
        best_target_per_prior.index_fill(0,best_prior_per_target_index,3.0)
        # 选取label
        labels=gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior<self.iou_threshold]=0.0
        boxes=gt_boxes[best_target_per_prior_index]
        center_form_boxes=self.corner_from_to_center(boxes)
        locations=self.boxes_from_to_locations(center_form_boxes,self.priors)
        # convert boxes to locations
        return locations,labels
    def boxes_from_to_locations(self,boxes,center_from_priors,size_variance,center_variance)->torch.Tensor:
        '''
        convert center_box to location
        Args:
            boxes:
            center_from_priors:
            size_variance:
            center_variance:
        Returns:
            locations
            torch.Tensor
        '''
        if boxes.dim()==(center_from_priors.dim()+1):
            center_from_priors=center_from_priors.unsqueeze(0)
        return torch.cat([
            (boxes[...,:2]-center_from_priors[...,:2])/center_from_priors[...,2:]/center_variance,
            torch.log(boxes[...,2:]/center_from_priors[...,2:])/size_variance
        ],dim=boxes.dim()-1)
    def decode(self,prediction):

        pass











            


