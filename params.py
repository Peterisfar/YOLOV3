# coding=utf-8

# project
DATA_PATH = "/home/leon/data/data/VOC"  # 数据集主路径
PROJECT_PATH = "/home/leon/doc/code/python_code/YOLOV3" # 项目路径
LOG_PATH = "/home/leon/doc/code/python_code/YOLOV3/log"
PRE_TRAIN_WEIGHT_PATH = "/home/leon/doc/code/python_code/YOLOV3/weights/darknet53.conv.74"


CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

# model
ANCHORS = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] # Anchors for big obj
STRIDES = [8, 16, 32]


# train


# test
input_shape = 544
batch_size = 1
nworkers = 0


conf_thresh = 0.01
nms_thresh = 0.5


