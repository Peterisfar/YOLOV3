# coding=utf-8
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
DATA_PATH = "/home/leon/data/data/VOC" # 数据集主路径
PROJECT_PATH = "/home/leon/doc/code/python_code/YOLOV3" # 项目路径
LOG_PATH = PROJECT_PATH + "/log"

# test
input_shape = 544
batch_size = 1
nworkers = 0
gpu = 0

conf_thresh = 0.01
nms_thresh = 0.5


