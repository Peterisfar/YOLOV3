from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.model import Darknet
from tqdm import tqdm
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import os
import params as pms
from utils.visualize import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


class Tester(object):
    def __init__(self,
                 cfg_path=None,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 visiual=None,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = pms.DATA["NUM"]
        self.__conf_threshold = pms.TEST["CONF_THRESH"]
        self.__nms_threshold = pms.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__visiual = visiual
        self.__eval = eval
        self.__classes = pms.DATA["CLASSES"]

        self.__model = Darknet(cfg_path=cfg_path, img_size=img_size).to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)


    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt['model'])
        print("loading weight file is done")
        del chkpt


    def test(self):
        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            for v in imgs:
                path = os.path.join(self.__visiual, v)
                print("test images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    path = os.path.join(pms.PROJECT_PATH, "data/{}".format(v))

                    cv2.imwrite(path, img)
                    print("saved images : {}".format(path))


        if self.__eval:
            mAP = 0
            print('*' * 20 + "Validate" + '*' * 20)

            with torch.no_grad():
                result = self.__evalter.APs_voc()
                for i in result:
                    print(i, result[i])
                    mAP += result[i]
                mAP = mAP / self.__num_class
                print('mAP:%g' % (mAP))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfg/yolov3-voc.cfg', help='cfg file path')
    parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    parser.add_argument('--visiual', type=str, default='./data/test', help='data augment flag')
    parser.add_argument('--eval', action='store_true', default=True, help='data augment flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester(cfg_path=opt.cfg_path,
            weight_path=opt.weight_path,
            gpu_id=opt.gpu_id,
            eval=opt.eval,
            visiual=opt.visiual).test()
