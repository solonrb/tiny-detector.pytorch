import easydict as Dict
# config for ssd priors
Priors_config=Dict()
Priors_config.prior_aspect_ratio=[
    [1/2,1/3],
    [1/2,1/3,1/4],
    [1/2,1/3,1/4],
    [1/2,1/3]
]
# default priors boxes size of every anchor layer
Priors_config.min_boxes_size=[
    [10, 16, 24],
    [32, 48],
    [64, 96],
    [128, 192, 256]
]
Priors_config.source_layers_index=[
    8,
    11,
    13
]
Priors_config.orign_image_size=[
    320,
    240
]
Priors_config.image_size_dict= img_size_dict = {
    128: [128, 96],
    160: [160, 120],
    320: [320, 240],
    480: [480, 360],
    640: [640, 480],
    1280: [1280, 960]
}
Priors_config.feature_map_shapes={
    128: [[16, 8, 4, 2], [12, 6, 3, 2]],
    160: [[20, 10, 5, 3], [15, 8, 4, 2]],
    320: [[40, 20, 10, 5], [30, 15, 8, 4]],
    480: [[60, 30, 15, 8], [45, 23, 12, 6]],
    640: [[80, 40, 20, 10], [60, 30, 15, 8]],
    1280: [[160, 80, 40, 20], [120, 60, 30, 15]]
}

# nms config
NMS_config=Dict()
NMS_config.iou_threshold=0.3
NMS_config.center_variance=0.1
NMS_config.size_variance=0.2


def update_config(image_size,):
    pass