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
        torch.clamp()
                    
                            


