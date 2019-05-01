from utils import VocDataset
from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.model import Darknet
from tqdm import tqdm
from model.loss import YoloV3Loss
from utils.tools import *
from eval.eval import *
import argparse


class Tester(object):
    def __init__(self,cfg_path=None,
                     weight_path=None,
                     gpu_id=0,
                     anno_file_type="test",
                     batch_size=1,
                     img_size=416,
                     num_workers=4,
                     iou_threshold=0.5,
                     conf_threshold=0.01,
                     nms_threshold=0.5,
                     model=None
                 ):
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.test_dataset = VocDataset(anno_file_type=anno_file_type,
                                            img_size=img_size,
                                            augment=False,
                                            mix_up=False
                                            )
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          collate_fn=self.test_dataset.collate_fn
                                          )
        self.__load__model(model, cfg_path, weight_path, gpu_id)
        self.criterion = YoloV3Loss()

    def __load__model(self, model, cfg_path, weight_path, gpu_id):
        if model is None:
            self.device = gpu.select_device(gpu_id)
            self.yolov3 = Darknet(cfg_path, self.img_size).to(self.device)
            if weight_path.endswith('.pt'):
                self.yolov3.load_state_dict(torch.load(weight_path, map_location=self.device)["model"])
            else:
                self.yolov3.load_darknet_weights(weight_path)
        else:
            self.device = next(model.parameters()).device
            self.yolov3 = model

    def test(self):
        self.yolov3.eval()
        seen = 0
        loss, precision, recall, f1, mp, mr, mAP, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class= [], [], []
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(tqdm(self.test_dataloader, desc='Computing mAp')):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                pred_de, pred = self.yolov3(imgs)

                loss_i , _= self.criterion(net=self.yolov3, p=pred, targets=targets)
                loss += loss_i.item()

                print(loss)
                # nms
                # output : 一个batch中每张图片预测值经过nms后剩下的boxes; shape : [[...], [...], ...]
                output = non_max_suppression(pred_de, conf_thres=self.conf_threshold, nms_thres=self.nms_threshold)

                for i, out in enumerate(output):
                    labels = targets[targets[:, -1] == i, :-1]  # 取出一个batch大小里第i张图片所有的标签
                    nl = len(labels)
                    tcls = labels[:, 4].tolist() if nl else [] # 所有标签对用的类别，shape为N*1
                    seen += 1
                    if out is None: # 预测box全部经nms过滤
                        if nl:
                            stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                            continue
                    else:
                        correct = [0] * len(out)
                        if nl:
                            detected = []
                            tbox = xywh2xyxy(labels[:, :4])

                            for j, (*out_box, out_conf, out_cls_conf, out_cls) in enumerate(out):
                                if len(detected) == nl:
                                    break

                                if out_cls.item() not in tcls:
                                    continue

                                iou, bi = bbox_iou(out_box, tbox).max(0)

                                if iou > self.iou_threshold and bi not in detected:
                                    correct[j] = 1
                                    detected.append(bi)
                            # stats : (correct, conf, cls, tcls)
                            stats.append((correct, out[:, 4].cpu(), out[:, 6].cpu(), tcls))

        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # zip(*)进行解压
        nt = np.bincount(stats[3].astype(np.int64), minlength=self.test_dataset.num_classes) # 统计target中每个类别出现的次数
        if len(stats):
            precision, recall, ap, f1, ap_class = ap_per_class(*stats)
            mp, mr, mAP, mf1 = precision.mean(), recall.mean(), ap.mean(), f1.mean()

        # print result
        pf = "%2s\t\t" + 'Images : %.3g\t\t' + 'Targets : %.3g\t\t' + 'P : %.3g\t\t' + \
            'R : %.3g\t\t' + 'mAP : %.3g\t\t' + 'F1 : %.3g\t\t'
        print(pf % ("All".ljust(10), seen, nt.sum(), mp, mr, mAP, mf1), end='\n\n')

        # print per class
        if len(stats):
            for i ,c in enumerate(ap_class):
                print(pf% (self.test_dataset.classes[c].ljust(10), seen, nt[c], precision[i], recall[i], ap[i], f1[i]))

        return mp, mr, mAP, mf1, loss / len(self.test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weight_path', type=str, default='weight/yolov3.weights', help='weight file path')
    parser.add_argument('--anno_file_type', type=str, default='test', help='data file type')
    parser.add_argument('--img_size', type=int, default=416, help='test image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of pytorch dataloader workers')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou threshold ')
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='threshold for object class confidence')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='threshold for nms')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester(cfg_path=opt.cfg_path,
            weight_path=opt.weight_path,
            anno_file_type=opt.anno_file_type,
            img_size=opt.img_size,
            num_workers=opt.num_workers,
            batch_size=opt.batch_size,
            iou_threshold=opt.iou_threshold,
            conf_threshold=opt.conf_threshold,
            nms_threshold=opt.nms_threshold,
            gpu_id=opt.gpu_id).test()
