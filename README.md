# YOLOV3
---
# Introduction
This is the branch of YOLOV3 about using model compression tricks.The dataset used is PASCAL VOC(not use difficulty). The eval tool is the voc2010. If you want to see the original code, you can switch to [master branch](https://github.com/Peterisfar/YOLOV3)

Subsequently, i will continue to update the code, involving new papers and tips.

---
## Results


| name | Train Dataset | Val Dataset | Params | Flops | Inference(CPU\|GPU) | mAP | notes |
| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----|
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 236M | - | - | 0.832 | darknet53 |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 27M | - | - | 0.792 | MobileNet-v2 & FPN(conv->dw+pw) |


`Note` : 

* YOLOV3-*-544 means test image size is 544. `"*"` means the multi-scale.
* In the test, the nms threshold is 0.5 and the conf_score is 0.01.
* Now only support the single gpu to train and test.

---
## Environment

* Nvida GeForce RTX 2080 Ti
* CUDA10.0
* CUDNN7.0
* ubuntu 16.04
* Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz

```bash
# install packages
pip3 install -r requirements.txt --user
```

---
## Brief

* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Step lr Schedule 
* [x] Multi-scale Training (320 to 640)
* [x] Focal loss
* [x] GIOU
* [x] Label smooth
* [x] Mixup
* [x] Cosine lr
* [x] MobileNet-V2


---
## Prepared work

### 1、Git clone YOLOV3 repository
```Bash
git clone https://github.com/Peterisfar/YOLOV3.git
```
update the `"PROJECT_PATH"` in the params.py.
### 2、Download dataset
* Download Pascal VOC dataset : [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). put them in the dir, and update the `"DATA_PATH"` in the params.py.
* Convert data format : Convert the pascal voc *.xml format to custom format (Image_path0 &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; xmin1,ymin1...)

```bash
cd YOLOV3 && git checkout model_compression && mkdir data
cd utils
python3 voc.py # get train_annotation.txt and test_annotation.txt in data/
```

### 3、Download weight file
* MobileNet-V2 pre-trained weight :  [mobilenetv2_1.0-0c6065bc.pth](https://pan.baidu.com/s/1BwObvtGalF2R2iE3u-XqhQ) 
* This repository test weight : [best_mobilenet_v2.pt](https://pan.baidu.com/s/1UHVSgqSg2OZhlcwAe-UJ8g)

Make dir `weight/` in the YOLOV3 and put the weight file in.

---
## Train

Run the following command to start training and see the details in the `config/yolov3_config_voc.py`

```Bash
WEIGHT_PATH=weight/mobilenetv2_1.0-0c6065bc.pth

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py --weight_path $WEIGHT_PATH --gpu_id 0 > nohup.log 2>&1 &

```

`Notes:`

* Training steps could run the `"cat nohup.log"` to print the log.
* It supports to resume training adding `--resume`, it will load `last.pt` automaticly.

---
## Test
You should define your weight file path `WEIGHT_FILE` and images file path `IMAGE_FILE`
```Bash
WEIGHT_PATH=weight/best_mobilenet_v2.pt
DATA_TEST=./data/test # your own images

CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_path $WEIGHT_PATH --gpu_id 0 --visiual $DATA_TEST --eval

```
The images can be seen in the `data/`

---
## TODO

* [ ] EfficientNet
* [ ] OctvConv

---
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* MobileNet-v2 : https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py

