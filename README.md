# YOLOV3
I tried to use yolov3 written by pytorch. The data set used is PASCAL VOC. Now the recurring effect is not the best. 
It may be caused by the loss function.  I will modify it later.And anyone also could give me some advice.
## Result
| name | Train Dataset | Val Dataset | mAP(others) | mAP(mine) | 
| :----- | :----- | :------ | :----- | :-----|
| YOLOV3-448<br>(darknet53.conv.74)</br> | 2007trainval + 2012trainval | 2007test | 0.79 | 0.631|

## Environment
* pytorch 1.0.0
* python 3.5.0
* numpy 1.14.5
* opencv-python 3.4.0.12

## Tools
* Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* Multiply Training (320 to 640)
* LR Schedule (LambdaLR)

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
* PyTorch `*.pt` format: https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI

## Train
```Bash
nohup python3 train.py > nohup.log 2>&1 &
```
## Test

## Reference
good recurrent
* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
