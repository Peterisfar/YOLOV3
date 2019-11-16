# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import torch
from torch.utils.data import Dataset, DataLoader
import params as pms
import cv2
import numpy as np
# from . import data_augment as dataAug
# from . import tools

import utils.data_augment as dataAug
import utils.tools as tools

class VocDataset(Dataset):
    def __init__(self, anno_file_type, img_size=416, augment=False, mix_up=True):
        self.img_size = img_size  # 图片尺寸，用于Multi-training or Multi-testing
        self.augment = augment
        self.classes = pms.CLASSES
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.mix_up = mix_up
        self.__annotations = self.__load_annotations(anno_file_type)

    def __len__(self):
        return  len(self.__annotations)

    def __getitem__(self, item):

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])

        # print(bboxes_org.shape)
        # print(bboxes_org)

        # if self.augment and self.mix_up:
        #     item_mix = random.randint(0, len(self.__annotations)-1)
        #     img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        #     img_org, bboxes_org = dataAug.MixupDetection()(np.copy(img_org), np.copy(bboxes_org),
        #                                           np.copy(img_mix), np.copy(bboxes_mix))
        # else:
        #     bboxes_org = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        img = img_org.transpose(2, 0, 1)

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes_org)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def __load_annotations(self, anno_type):
        """加载annotation.txt中所有标签文件"""
        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(pms.PROJECT_PATH, 'data', anno_type+"_annotation.txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x:len(x)>0, f.readlines()))
        assert len(annotations)>0, "No images found in {}".format(anno_path)

        return annotations

    def __parse_annotation(self, annotation):
        """
        读取annotation中image_path对应的图片，并将该图片进行resize(不改变图片的高宽比)
        获取annotation中所有的bbox，并将这些bbox的坐标(xmin, ymin, xmax, ymax)进行纠正，
        使得纠正后bbox在resize后的图片中的相对位置与纠正前bbox在resize前的图片中的相对位置相同
        :param annotation: 图片地址和bbox的坐标、类别，
        如：image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...
        :return: image和bboxes
        bboxes的shape为(N, 5)，其中N表示一站图中有N个bbox，5表示(xmin, ymin, xmax, ymax, class_ind)
        """
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])

        if self.augment:
            img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        else: # inference
            img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):
        """
        标签分配.对于单张图片所有的GT框bboxes分配anchor.
        (1)顺序选取一个bbox,转换其坐标为xyxy2xywh；并且按各检测分支的尺度对bbox xywh进行缩放
        (2)依次将bbox与每一检测层的anchors进行iou的计算，选择iou最大的anchor来负责该bbox。
        若所有检测层的所有iou均小于阈值则从所有检测层中选择最大iou对应的anchor负责检测它。

        注意：1、同一个GT可能会分配给多个anchor,这些anchor有可能在同一层，也有可能在不同的层
            2、bbox的总数量可能会比实际多，因为同一个GT可能会分配给多层检测层。
        :param img: 输入图像，将其归一化到[0-1]
        :param bboxes: bboxes为该图所以的GT，维度为[N, 5]其中 将其展开为[xmin, ymin, xmax, ymax, cls]
        :return: img, label
        """

        ANCHORS = np.array(pms.ANCHORS)
        STRIDES = np.array(pms.STRIDES)
        TRAIN_OUTPUT_SIZE = self.img_size / STRIDES
        ANCHOR_PER_SCALE = 3

        label = [np.zeros((int(TRAIN_OUTPUT_SIZE[i]), int(TRAIN_OUTPUT_SIZE[i]), ANCHOR_PER_SCALE, 5+self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0

            # 将xyxy 转为xywh
            # bbox_xywh = tools.xyxy2xywh(bbox_coor.reshape(1, -1))
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            # print("bbox_xywh: ", bbox_xywh)

            # 分别得到三个检测分支的xywh
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / STRIDES[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((ANCHOR_PER_SCALE, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 用于补偿
                anchors_xywh[:, 2:4] = ANCHORS[i]

                iou_scale = tools.iou_calc2(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = one_hot

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)  # 所有检测层的iou
                best_detect = int(best_anchor_ind / ANCHOR_PER_SCALE)
                best_anchor = int(best_anchor_ind % ANCHOR_PER_SCALE)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = one_hot

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh


        # print("num of sbboxes is : ", bbox_count[0])
        # print("num of mbboxes is : ", bbox_count[1])
        # print("num of lbboxes is : ", bbox_count[2])

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == "__main__":
    voc_dataset = VocDataset(anno_file_type="train", img_size=448, augment=True, mix_up=True)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)


    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i==0:
            print(img.shape)
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape)
            print(sbboxes.shape)
            print(mbboxes.shape)
            print(lbboxes.shape)

            if img.shape[0] == 1:
                labels = np.concatenate([label_sbbox.reshape(-1, 25), label_mbbox.reshape(-1, 25), label_lbbox.reshape(-1, 25)], axis=0)
                labels_mask = labels[..., 4]>0
                labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 5:], axis=-1).reshape(-1, 1)], axis=-1)

                print(labels.shape)
                tools.plot_box(labels, img, id=1)
