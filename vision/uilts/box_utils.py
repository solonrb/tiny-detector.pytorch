import math
import torch


def GenerateAnchor(feature_map_list,image_size,min_boxes,clamp=True,aspect_ratio=None)->torch.Tensor:
    '''
    args:
        feature_map_list: the shape of source layers
        image_size: the shape of input image size (w,h)
        min_boxes:the min prior box size for source layers
        clamp: clamp output in (0,1)
        aspect_ratio: the scale for prior box in source layers
    return:
        torch.Tensor:Priors ----->(num_priors,4)
    '''
    priors=[]
    for index in range(len(0,feature_map_list[0])):
        feature_map_w=feature_map_list[0][index]
        feature_map_h=feature_map_list[1][index]
        for i in range(0,feature_map_w):
            for j in range(0,feature_map_h):
                center_x=(i+0.5)/feature_map_w
                center_y=(j+0.5)/feature_map_h
                for wh in min_boxes[index]:
                    w=wh/image_size[0]
                    h=wh/image_size[1]
                    boxes1=[center_x,center_y,w,h] 
                    priors.append(boxes1)
                    if aspect_ratio is not None:
                        for ratio in aspect_ratio[index]:
                            ws=wh*math.sqrt(ratio)/image_size[0]
                            hs=wh/math.sqrt(ratio)/image_size[1]
                            ws2=wh/math.sqrt(ratio)/image_size[0]
                            hs2=wh*math.sqrt(ratio)/image_size[1]
                            priors.append([center_x,center_y,ws,hs])
                            priors.append([center_x,center_y,ws2,hs2])
    priors=torch.tensor(priors)
    if clamp:
        priors=torch.clamp(priors,min=0.0,max=1.0)
    return priors

def area_of(left_up,bottom_down)->torch.Tensor:
    wh=torch.clamp(bottom_down-left_up,min=0.0)
    return wh[...,0]*wh[...,1]
def iou(boxes1,boxes2,eps=1e-5)->torch.Tensor:
    '''
    cal iou of two boxes
    Args:
        boxes1:
        boxes2:
        eps:

    Returns:
        torch.Tensor
    '''
    overlap_top_left=torch.max(boxes1[...,:2],boxes2[...,:2])
    overlap_right_bottom=torch.min(boxes1[...,2:],boxes2[...,2:])
    area_overlap=area_of(overlap_top_left,overlap_right_bottom)
    area_boxes1=area_of(boxes1[...,:2],boxes1[...,2:])
    area_boxes2=area_of(boxes2[...,:2],boxes2[...,2:])
    iou=area_overlap/(area_boxes1+area_boxes2-area_overlap+eps)
    return iou
def nms(boxes_scores,threshold,top_k=-1,truncated_num=200):
    '''

    Args:
       boxes_scores: shape of [N,5]--->(x,y,w,h,s)
       threshold: iou-threshold
       top_k: return the result truncated with k numbers
       truncated_num: select truncated number boxes for nms
    Returns:
        select boxes_scores [n,5]
    '''
    boxes=boxes_scores[...,:4]
    scores=boxes_scores[...,4:]
    # 降序排序
    _,sorted_index=scores.sort(descending=True)
    sorted_index=sorted_index[:truncated_num]
    keep=[]
    while sorted_index.numel()>0:
        if sorted_index.numel()==1:
            keep.append(sorted_index[0].item())
            break
        else:
            current=sorted_index[0]
            keep.append(current.item())
            current_box=boxes[current,:]
            sorted_index=sorted_index[1:]
            reset_boxes=boxes_scores[sorted_index,:]
            # 计算iou
            c_iou_scores=iou(reset_boxes,current_box.unsqueeze(0))
            # 根据iou挑选该留待下轮迭代的框的index
            sorted_index=sorted_index[c_iou_scores<threshold]
    if top_k:
        keep=keep[:top_k]
    return boxes_scores[keep,:]
def soft_nms(boxes_scores,score_threshold,simga=0.5):
    '''
    implementation of soft-nms
    Args:
        boxes_scores: shape of [n,5]--->(x,y,w,h,s)
        score_threshold: score threshold
        simga: scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
    Returns:
        boxes-scores list or tensor
    '''
    picked_boxes_scores=[]
    while boxes_scores.numel()>0:
        if boxes_scores.numel()==1:
            picked_boxes_scores.append(boxes_scores[0])
        else:
            _,max_score_index=boxes_scores[:,-1].sort(descending=True)
            max_boxes_scores=boxes_scores[max_score_index[0],:]
            picked_boxes_scores.append(max_boxes_scores)
            current_box=max_boxes_scores[:,:-1]
            boxes_scores[max_score_index[0],:]=boxes_scores[-1,:]
            #将得分最大框从集合中去掉 首尾赋值
            boxes_scores=boxes_scores[:-1,:]
            reset_boxes=boxes_scores[:,:-1]
            c_iou=iou(reset_boxes,current_box)
            boxes_scores[:,-1]=boxes_scores[:,-1]*torch.exp(-torch.pow(c_iou,2)/simga)
            mask=boxes_scores[:,-1]>score_threshold
            boxes_scores=boxes_scores[mask,:]
    if len(picked_boxes_scores)>0:
        return torch.stack(picked_boxes_scores)
    else:
        return torch.tensor([])

def hard_negative_mining(loss,pos_negative_ratio,labels):
    '''
    针对每个样本的难样本挖掘
    Args:
        loss:
        pos_negative_ratio:
        labels:

    Returns:

    '''
    pos_mask=labels>0
    pos_num=pos_mask.long().sum(dim=1,keepdim=True)
    num_neg=pos_num*pos_negative_ratio
    loss[pos_mask]=-math.inf
    _,index=loss.sort(dim=1,decrending=True)
    _,order=index.sort(dim=1)
    neg_mask=order<num_neg
    return pos_mask | neg_mask








                    
                            


