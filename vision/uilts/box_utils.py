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
                            ws2=wh/math.sqrt(ratio)/image_size[1]
                            hs2=wh*math.sqrt(ratio)/image_size[0]
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
def soft_nms(boxes_scores,score_threshold,iou_threshold,top_k=-1):
    '''
    implementation of soft_nms
    Args:
        boxes_scores:
        score_threshold:
        iou_threshold:
    Returns:
        top_k boxes
    '''




                    
                            


