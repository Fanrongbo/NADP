import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np
from option.config import cfg
from proDAdata.randaugment import RandAugmentMC
from proDAdata.augmentations import *
normMean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
normStd = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def standardization(image):
    image = ((image / 255) - normMean) / normStd
    return image

class SegmentationMMLabDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.randaug = RandAugmentMC(2, 3)
        self.augmentations=Compose([RandomSized(opt.resize),
                    RandomCrop(opt.rcrop),
                    RandomHorizontallyFlip(opt.hflip)])
        self.opt = opt
        self.root = opt.dataroot
        self.img_size=(512,512)
        ### input T1_img
        if opt.phase in ['train','val']:
            self.img=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'img_dir',opt.phase)
            self.imgpath=sorted(make_dataset([self.img]))

            self.dir_label=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'ann_dir',opt.phase)
            self.label_paths = sorted(make_dataset([self.dir_label]))
        elif opt.phase in ['targetSelect']:
            self.img = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'img_dir', 'train')
            print('opt.hardspilt',opt.hardspilt)
            list_path = opt.hardspilt.format('all')
            print('list_path',list_path)
            ### input T2_img
            # self.t2_paths = sorted(make_dataset([self.dir_t2]))
            # dir_label = 'label'
            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'ann_dir', 'train')

            self.imgpath=[]
            self.label_paths=[]
            with open(list_path) as f:
                for i_id in f:
                    i_id=i_id.strip()
                    self.imgpath.append(i_id)
                    self.label_paths.append(i_id.replace('img_dir', 'ann_dir'))
            print('label_paths', self.label_paths)
        elif opt.phase in ['target','targetTest']:
            self.img = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'img_dir', 'train')
            self.imgpath = sorted(make_dataset([self.img]))

            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'ann_dir', 'train')
            self.label_paths = sorted(make_dataset([self.dir_label]))


        self.dataset_size = len(self.label_paths)
        self.preprocess = Preprocessing(
            img_size=self.opt.img_size,
            with_random_hflip=opt.aug,
            with_random_vflip=opt.aug,
            with_scale_random_crop=opt.aug,
            with_random_blur=opt.aug,
            with_Lchannel=False
        )
    def __getitem__(self, index):


        ### input T1_img
        img_path = self.imgpath[index]
        label_path = self.label_paths[index]

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(label_path)
        # img = img.resize(self.img_size, Image.BILINEAR)
        # lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        # print('lbl',lbl.shape,lbl.max(),lbl.min())
        # lbl[lbl == 0] = 6  # 首先将0变为-1，以避免与其他值冲突
        # lbl[lbl == 5] = 0  # 将5变为0
        # lbl[lbl == 6] = 5  # 最后将-1变为5
        # lbl = np.array(lbl, dtype=np.uint8)

        lblori=lbl.copy()
        # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img_full = img.copy().astype(np.float64)
        # img_full -= self.mean
        [imgA], [lblA] = self.preprocess.transform([img], [lbl], to_tensor=False)
        imgA = np.array(imgA)
        imgA = imgA.transpose(2, 0, 1)

        imgA = imgA.astype(float)
        lblA = np.array(lblA)

        # img_full = img_full.astype(float) / 255.0
        # img_full=standardization(img_full.astype(float) )
        img_full = img_full.transpose(2, 0, 1)
        lp, lpsoft, weak_params = None, None, None

        input_dict = {}
        if self.opt.augmentations:
        # if self.augmentations != None:
            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))
            # img_strong, _, _ = self.transform(img_strong, lbl)
            img_strong = np.array(img_strong)
            input_dict['img_strong'] = torch.from_numpy(img_strong.transpose(2, 0, 1)).float()/ 255.0
            input_dict['params'] = params

        # img, lbl_, lp = self.transform(img, lbl, lp)

        input_dict['img'] = torch.from_numpy(imgA).float()/ 255.0
        input_dict['label'] = self.transformlabel(lblA)

        input_dict['img_full'] = torch.from_numpy(img_full).float()/ 255.0
        input_dict['label_full'] = self.transformlabel(lblori)

        input_dict['label_path'] = label_path
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  # full2weak
        input_dict['img_path'] = img_path
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        return input_dict


    def __len__(self):
        return len(self.label_paths) // self.opt.batch_size * self.opt.batch_size
    def transformlabel(self,lbl):
        classes = np.unique(lbl)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones
        lbl = torch.from_numpy(lbl).long()
        return lbl
    def transform(self, img, lbl, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        # img = img.astype(np.float64)
        # img -= self.mean
        # img = img.astype(float) / 255.0
        # img=standardization(img.astype(float))

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones

        # if check and not np.all(
        #         np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):  # todo: understanding the meaning
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)
            # if not np.all(np.unique(lp[lp != self.ignore_index]) < self.n_classes):
            #     raise ValueError("lp Segmentation map contained invalid class values")
            lp = torch.from_numpy(lp).long()

        return img, lbl, lp

    def encode_segmap(self, mask):
        # Put all void classes to zero
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy