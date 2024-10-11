import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np
from option.config import cfg


class SegmentationMMLabDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ### input T1_img
        if opt.phase in ['train','val']:
            self.img=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'img_dir',opt.phase)
            self.imgpath=sorted(make_dataset([self.img]))

            self.dir_label=os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.s],'ann_dir',opt.phase)
            self.label_paths = sorted(make_dataset([self.dir_label]))

        else:
            self.img = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'img_dir', 'train')
            self.imgpath = sorted(make_dataset([self.img]))

            self.dir_label = os.path.join(opt.dataroot, cfg.TRAINLOG.DATA_NAMES[opt.t], 'ann_dir', 'train')
            self.label_paths = sorted(make_dataset([self.dir_label]))


        self.dataset_size = len(self.label_paths)
        if self.opt.phase == 'train':
            print('with_Lchannel=opt.LChannel', opt.LChannel)
            self.preprocess = Preprocessing(
                img_size=self.opt.img_size,
                with_random_hflip=opt.aug,
                with_random_vflip=opt.aug,
                with_scale_random_crop=opt.aug,
                with_random_blur=opt.aug,
                with_Lchannel=opt.LChannel
            )
        else:
            self.preprocess= Preprocessing(
                img_size=self.opt.img_size,
                with_Lchannel=opt.LChannel
                )
        # self.preprocess = Preprocessing(
        #     img_size=self.opt.img_size,
        #     with_Lchannel=opt.LChannel
        # )
    def __getitem__(self, index):
        ### input T1_img 
        imgpath = self.imgpath[index]
        img = np.asarray(Image.open(imgpath).convert('RGB'))
        # print(t1_img.shape)
        if img.shape[0]<self.opt.img_size or img.shape[1]<self.opt.img_size:
            img = np.resize(img, (self.opt.img_size, self.opt.img_size,3))
        ### input label
        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path), dtype=np.uint8)
        # print('label',label.shape)
        if len(label.shape)==3:
            label=label[:,:,0]
        if label.shape[0] < self.opt.img_size or label.shape[1]<self.opt.img_size:
            label = np.resize(label, (self.opt.img_size, self.opt.img_size))

        # if self.opt.label_norm == True:
        #     label = label // 255
        # print(t1_path)
        ### transform
        [tensor], [label_tensor] = self.preprocess.transform([img], [label], to_tensor=True)
        img_full, label_full, _ = self.transform(img, label, None)

        input_dict = {'img_full':img_full,'label_full':label_full,'img': tensor, 'label': label_tensor, 'imgpath': imgpath, 'label_path': label_path}

        return input_dict

    def __len__(self):
        return len(self.label_paths) // self.opt.batch_size * self.opt.batch_size
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
        img = img.astype(np.float64)
        # img -= self.mean
        img = img.astype(float) / 255.0
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
