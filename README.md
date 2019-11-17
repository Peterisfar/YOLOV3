# YOLOV3
---
# Introduction

This is my own YOLOV3 written in pytorch, and is also the first time i have reproduced a object detection model.The dataset used is PASCAL VOC(not use difficulty). The eval tool is the voc2010. Now the mAP gains the goal score.

Subsequently, i will continue to update the code to make it more concise , and add the new and efficient tricks.

---
## Results


| name | Train Dataset | Val Dataset | mAP(others) | mAP(mine) | notes |
| :----- | :----- | :------ | :----- | :-----| :-----|
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 0.769 | 0.768 | baseline(augument + step lr) |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.793  | 0.803 | \+multi-scale training |
  
  
`Note` : 

* YOLOV3-448-544 means train image size is 448 and test image size is 544. `"*"` means the multi-scale.
* In the test, the nms threshold is 0.5 and the conf_score is 0.01.
* Now only support the single gpu to train and test.

---
## Environment

* Nvida GeForce RTX 2080 Ti
* CUDA10.0
* CUDNN7.0
* ubuntu 16.04

```bash
# install packages
pip3 install -r requirements.txt --user
```

---
## Brief

* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Step lr Schedule 
* [x] Multi-scale Training (320 to 640)
* [ ] Mixup
* [ ] Label smooth
* [ ] GIOU
* [ ] focal loss

---
## Prepared work

### 1、Git clone YOLOV3 repository
```Bash
git clone https://github.com/Peterisfar/YOLOV3.git
```
### 2、Download dataset
* Download Pascal VOC dataset : [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* Convert data format : Convert the pascal voc *.xml format to custom format (Image_path0 &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; Image_path1 xmin1,ymin1...)

```bash
cd YOLOV3 && mkdir data
cd utils
python3 voc.py # get train_annotation.txt and test_annotation.txt in data/
```

### 3、Download pre-weight 
* Darknet pre-trained weight :  [darknet53-448.weights](https://pjreddie.com/media/files/darknet53_448.weights) 
* This repository test weight : [best.pt](https://pan.baidu.com/s/1wQgaBe81-OPm0YlbZFR_Kw)

Make dir `weight/` in the YOLOV3 and put the weight file in.

---
## Train

Run the following command to start training and see the details in the `params.py`

```Bash
CFG_PATH=cfg/yolov3-voc.cfg
WEIGHT_PATH=weight

nohup CUDA_VISIABLE_DEVICES=0 python3 train.py --cfg_path $CFG_PATH --weight_path $WEIGHT_PATH --gpu_id 0 > nohup.log 2>&1 &

```

`Notes:`

* Training steps could run the `"cat nohup.log"` to print the log.
* It supports to resume training adding `--resume`, it will load `last.pt` automaticly.

---
## Test

Run the follwing command to test, 

```Bash
CFG_PATH=cfg/yolov3-voc.cfg
WEIGHT_PATH=weight
DATA_TEST=./data/test # your own images

nohup CUDA_VISIABLE_DEVICES=0 python3 test.py --cfg_path $CFG_PATH --weight_path $WEIGHT_PATH --gpu_id 0 --visiual $DATA_TEST --eval> nohup.log 2>&1 &

```
---
## TODO

* [ ] Mish
* [ ] OctvConv
* [ ] Mobilenet v1-v3

---
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
