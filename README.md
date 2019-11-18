# YOLOV3
This is my own YOLOV3 written in pytorch, and is also the first time i have reproduced a object detection model.The dataset used is PASCAL VOC(not use difficulty). Now the mAP gains the goal.


## Result
| name | Train Dataset | Val Dataset | mAP(others) | mAP(mine) | note | 
| :----- | :----- | :------ | :----- | :-----|
| YOLOV3-448<br>(darknet53.conv.74)</br> | 2007trainval + 2012trainval | 2007test | 0.769 | 0.768 | augument + step lr |

## Environment
* pytorch 1.0.0
* python 3.5.0
* numpy 1.14.5
* opencv-python 3.4.0.12

## Tools
* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] LR Schedule 
* [ ] Multi-scale Training (320 to 640)
* [ ] MIXUP
* [ ] Label smooth
* [ ] GIOU
* [ ] focal loss

## Prepared
### 1、Git clone YOLOV3 repository 
```Bash
git clone https://github.com/Peterisfar/YOLOV3.git
```
### 2、Download dataset
* Download [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
       、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
       、[VOC 2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* convert voc data type to custom data type
custom datatype is : Image name &nbsp; xmin,ymin,xmax,ymax &nbsp; \[...]
```bash
cd utils
python3 voc.py // get train_annotation.txt and test_annotation.txt
```
### 3、Download pre-weight 
* Darknet `*.weights` format: https://pjreddie.com/media/files/yolov3.weights

## Train
```Bash
nohup python3 train.py > nohup.log 2>&1 &
```
## Test
You should define your weight file path `WEIGHT_FILE` and images file path `IMAGE_FILE`
```Bash

WEIGHT_FILE=weight/best.pt
IMAGE_FILE=./data/test

CUDA_VISIABLE_DEVICES=0 python3 test.py --cfg_path cfg/yolov3-voc.cfg --weight_path $WEIGHT_FILE  --visiual $IMAGE_FILE --gpu_id 0 

```

## Reference
good recurrent
* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
