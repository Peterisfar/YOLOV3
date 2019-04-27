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
import random
from . import data_augment as dataAug
from . import tools


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

        if self.augment and self.mix_up:
            item_mix = random.randint(0, len(self.__annotations)-1)
            img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
            img_org, bboxes_org = dataAug.MixupDetection()(np.copy(img_org), np.copy(bboxes_org),
                                                  np.copy(img_mix), np.copy(bboxes_mix))
        else:
            bboxes_org = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        img, label = self.__creat_label(img_org, bboxes_org)

        # img [H, W, C] ---> [C, H, W]
        return torch.from_numpy(img), torch.from_numpy(label).float()

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
            img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes)) #  resize [img_size, img_size]
        else: # inference
            img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def __creat_label(self, img, bboxes):
        """
        :param img: 输入图像，将其归一化到[0-1]
        :param bboxes: bboxes[N, 6] : [N*[xmin, ymin, xmax, ymax, cls, multiup_w]] 转换为[N*[xcen, ycen, w, h, cls, multiup_w]] 并将其归一化
        :return: img, label
        """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0  # 图片归一化
        bboxes[:, :4] = tools.xyxy2xywh(bboxes[:, :4]) / self.img_size  # 坐标转化以及归一化

        return img, bboxes

    @staticmethod
    def collate_fn(batch):
        """collate_fn用于分配label box到指定的batch,
           labels shape : [bs*N, 7]"""
        imgs, labels = list(zip(*batch))
        # label new shape : N * [x, y, w, h, cls, mixup_weight, bs_id]
        labels = [torch.cat((l, torch.full((len(l), 1), 0.0)), -1) for l in labels]
        # labels input shape : bs * [ N, 6] -----[x, y, w, h, cls, mixup_weight]
        for i, l in enumerate(labels):
            l[:, -1] = i
        return torch.stack(imgs, 0), torch.cat(labels, 0)


# if __name__ == "__main__":
#     # test dataset is ok!
#     voc_dataset = VocDataset(anno_file_type="train", img_size=640, augment=True, mix_up=True)
#     dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0, collate_fn=voc_dataset.collate_fn)
#
#     dataiter = iter(dataloader)
#     img, bboxes = next(dataiter) # img : [bs, C, H, W]
#                                  # bboxes : [bs*N, 7]
#     tools.plot_box(bboxes, img)
