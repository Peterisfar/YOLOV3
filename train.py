import logging
import utils.gpu as gpu
from model.model import Darknet
from model.loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, cfg_path,
                        weight_path, # 权重文件路径
                       anno_file_type,
                        img_size,
                        resume,  # 继续训练
                        epochs,
                        batch_size,
                        multi_scale_train,
                         num_works,
                        augment, # 数据增强
                        lr_init,
                        lr_end,
                        warm_up_epoch,
                        mix_up,
                        momentum,
                        weight_decay,
                        focal_loss,
                        iou_threshold_loss,
                        gpu_id
                 ):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.warmup_epoch = warm_up_epoch
        self.best_mAP = 0.
        self.epochs = epochs
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.weight_path = weight_path
        self.multi_scale_train = multi_scale_train

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
                                           )
        self.yolov3 = Darknet(cfg_path=cfg_path, img_size=img_size).to(self.device)
        self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(),
                                   lr=lr_init,
                                   momentum=momentum,
                                   weight_decay=weight_decay
                                   )
        # self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(focal_loss=focal_loss,
                                    iou_threshold_loss=iou_threshold_loss)
        self.__load_model_weights(weight_path, resume)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 40], gamma=0.1,
                                                  last_epoch=self.start_epoch - 1)


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
            pre_trained_weight = os.path.join(weight_path, "darknet53_448.weights")
            self.yolov3.load_darknet_weights(pre_trained_weight)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
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
            torch.save(chkpt, os.path.join(self.weight_path, 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))


        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            self.scheduler.step()
            # self.optimizer.zero_grad()

            mloss = torch.zeros(4).to(self.device)  # mean losses
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_xywh, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_xywh, loss_conf, loss_cls, loss]).to(self.device)
                mloss = (mloss * i + loss_items) / (i + 1) # 平均值

                # Print batch results
                if i%10==0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_xywh: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0
            if epoch >= 20:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    result = Evaluator(self.yolov3).APs_voc()

                    for i in result:
                        print(i, result[i])
                        mAP += result[i]
                    mAP = mAP/self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            self.__save_model_weights(epoch, mAP)  # TODO:BUG CUDA out of memory
            print('best mAP : %g' % (self.best_mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weight_path', type=str, default='weight', help='weight file path')
    parser.add_argument('--anno_file_type', type=str, default='train', help='data file type')
    parser.add_argument('--img_size', type=int, default=448, help='image size')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--multi_scale_train', action='store_false', default=False, help='multi scale train flag')
    parser.add_argument('--num_works', type=int, default=4, help='number of pytorch dataloader workers') # Bug
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
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
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
            gpu_id=opt.gpu_id).train()
