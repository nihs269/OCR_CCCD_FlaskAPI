opt = {
    "img-size": 640,
    "conf-thres": 0.4,
    "iou-thres": 0.15,
    "device": 'cpu',
}

scan_opt = {
    "img-size": 640,  # default image size
    "conf-thres": 0.4,  # confidence threshold for inference.
    "iou-thres": 0.15,  # NMS IoU threshold for inference.
    "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
}
