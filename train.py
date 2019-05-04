import logging
import utils.gpu as gpu
from model.model import Darknet
from model.loss import YoloV3Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import os
import time
import random
import argparse
from eval.evaluator import *


class Trainer(object):
    def __init__(self, cfg_path,
                        weight_path, # 权重文件路径
                       anno_file_type="train",
                        img_size=416,
                        resume=False,  # 继续训练
                        epochs=300,
                        batch_size=8,
                        multi_scale_train=True,
                         num_works=4,
                        augment=True, # 数据增强
                        lr_init=0.001,
                        lr_end=10e-6,
                        warm_up_epoch=2,
                        mix_up=False,
                        momentum=0.9,
                        weight_decay=0.005,
                        focal_loss=True,
                        iou_threshold_loss=0.5,
                        label_smoothing=False,
                        conf_threshold=0.001,
                        nms_threshold=0.5,
                        gpu_id=0
                 ):
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.warmup_epoch = warm_up_epoch
        self.best_mAP = 0.
        self.epochs = epochs
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.weight_path = weight_path
        self.multi_scale_train = multi_scale_train
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.iou_threshold_loss = iou_threshold_loss
        self.train_dataset = data.VocDataset(anno_file_type=anno_file_type,
                                              img_size=img_size,
                                              augment=augment,
                                              mix_up=mix_up
                                              )
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_works,
                                           shuffle=True,
                                           sampler=None,  # 可以用于分布式训练
                                           collate_fn=self.train_dataset.collate_fn
                                           )
        self.yolov3 = Darknet(cfg_path=cfg_path, img_size=img_size).to(self.device)
        self.__load_model_weights(weight_path, resume)

        self.optimizer = optim.SGD(self.yolov3.parameters(),
                                   lr=lr_init,
                                   momentum=momentum,
                                   weight_decay=weight_decay
                                   )

        self.criterion = YoloV3Loss(focal_loss=focal_loss,
                                    iou_threshold_loss=iou_threshold_loss,
                                    label_smoothing=label_smoothing,
                                    )


    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(weight_path, "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.yolov3.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            pre_trained_weight = os.path.join(weight_path, "darknet53.conv.74")
            self.yolov3.load_darknet_weights(pre_trained_weight)

    def __save_model_weights(self, mAP, epoch):
        best_weight = os.path.join(self.weight_path, "best.pt")
        last_weight = os.path.join(self.weight_path, "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt, best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(self.weight_path, 'backup%g.pt'%epoch))
        del chkpt

    def train(self):
        t = time.time()
        nb = len(self.train_dataloader)
        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()
            self.optimizer.zero_grad()
            print(('\n%8s%12s' + '%10s' * 7)
                  % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))
            mloss = torch.zeros(5).to(self.device)  # mean losses
            for i, (imgs, targets) in enumerate(self.train_dataloader):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                nt = len(targets)

                pred = self.yolov3(imgs)

                loss, loss_items = self.criterion(net=self.yolov3, p=pred, targets=targets)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update run       ning mean of tracked metrics
                mloss = (mloss * i + loss_items) / (i + 1) # 平均值

                # Print batch results
                s = ('%8s%12s' + '%10.3g' * 7) % (
                    '%g/%g' % (epoch, self.epochs - 1),
                    '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
                print(s)
                t = time.time()

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))
            if epoch > 0:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    result = Evaluator(self.yolov3).APs_voc()
                    mAP = 0
                    for i in result:
                        print(i, result[i])
                        mAP += result[i]
                    print('mAP:%g'%(mAP/self.train_dataset.num_classes))

                self.__save_model_weights(mAP, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weight_path', type=str, default='weight', help='weight file path')
    parser.add_argument('--anno_file_type', type=str, default='train', help='data file type')
    parser.add_argument('--img_size', type=int, default=416, help='image size')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--multi_scale_train', action='store_false', default=False, help='multi scale train flag')
    parser.add_argument('--num_works', type=int, default=0, help='number of pytorch dataloader workers')
    parser.add_argument('--augment', action='store_false', default=True, help='data augment flag')
    parser.add_argument('--lr_init', type=float, default=0.0001, help='learning rate at start')
    parser.add_argument('--lr_end', type=float, default=10e-6, help='learning rate at end')
    parser.add_argument('--warm_up_epoch', type=int, default=2, help='the epochs for lr warm up')
    parser.add_argument('--mix_up', action='store_false', default=False, help='mix up flag')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--focal_loss', action='store_false', default=False, help='focal loss flag')
    parser.add_argument('--iou_threshold_loss', type=float, default=0.5, help='iou threshold in calculate loss')
    parser.add_argument('--label_smoothing', action='store_false', default=False, help='label smoothing flag')
    parser.add_argument('--conf_threshold', type=float, default=0.005, help='threshold for object class confidence')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='threshold for nms')
    parser.add_argument('--gpu_id', type=int, default=3, help='gpu id')
    opt = parser.parse_args()

    Trainer(cfg_path=opt.cfg_path,
            weight_path=opt.weight_path,
            anno_file_type=opt.anno_file_type,
            img_size=opt.img_size,
            resume=opt.resume,
            epochs=opt.epochs,
            batch_size=opt.batch_size,
            multi_scale_train=opt.multi_scale_train,
            num_works=opt.num_works,
            augment=opt.augment,
            lr_init=opt.lr_init,
            lr_end=opt.lr_end,
            warm_up_epoch=opt.warm_up_epoch,
            mix_up=opt.mix_up,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            focal_loss=opt.focal_loss,
            iou_threshold_loss=opt.iou_threshold_loss,
            label_smoothing=opt.label_smoothing,
            conf_threshold=opt.conf_threshold,
            nms_threshold=opt.nms_threshold,
            gpu_id=opt.gpu_id).train()