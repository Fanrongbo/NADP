import os.path
import torch
from data.image_folder import make_dataset
from data.preprocessing import Preprocessing
from PIL import Image
import numpy as np
from option.config import cfg
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
# from albumentations.pytorch import ToTensorV2
# import torchvision.transforms as T
import random


def get_transforms(weak_params):
    """基于weak_params生成Albumentations变换列表"""
    transforms_list = [
        A.Rotate(limit=(weak_params['angle'], weak_params['angle']), p=1),
        A.ShiftScaleRotate(shift_limit=(weak_params['translate'][0] / 512, weak_params['translate'][1] / 512),
                           scale_limit=(weak_params['scale'] - 1, weak_params['scale'] - 1),
                           rotate_limit=0, p=1, border_mode=0),
        A.Affine(shear=(weak_params['shear'], weak_params['shear']), p=1),
    ]

    if weak_params['flip']:
        transforms_list.append(A.HorizontalFlip(p=1))

    return A.Compose(transforms_list, additional_targets={'label': 'image'})


def transform(image, label, weak_params):
    """应用由weak_params指定的变换到图像和标签上"""
    transforms = get_transforms(weak_params)
    transformed = transforms(image=image, label=label)
    return transformed['image'], transformed['label']
class SegmentationMMLabDataset(torch.utils.data.Dataset):

    def initialize(self, opt):
        self.transform = transform
        # self.random_affine = RandomAffineWrapper(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10)
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


    def get_affine_matrix(self,weak_params):
        """根据weak_params和特征图的尺寸计算仿射矩阵"""
        angle = np.radians(weak_params['angle'])  # 将角度从度转换为弧度
        translate = weak_params['translate']
        scale = weak_params['scale']
        shear = np.radians(weak_params['shear'])

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        tx, ty = translate[0] / self.opt.img_size, translate[1] / self.opt.img_size
        sx, sy = scale, scale
        shx, shy = np.tan(shear), np.tan(shear)

        matrix = np.array([
            [cos_a * sx, -sin_a * sy + shy, tx],
            [sin_a * sx + shx, cos_a * sy, ty]
        ])
        if weak_params['flip']:
            matrix[:, 0] *= -1  # 如果需要翻转，则在x轴方向上取反

        # 返回2x3的仿射矩阵
        return matrix.astype(np.float32)
    def __getitem__(self, index):
        ### input T1_img 
        imgpath = self.imgpath[index]
        # img = np.asarray(Image.open(imgpath).convert('RGB'))
        img = np.asarray(cv2.imread(imgpath))

        # print(t1_img.shape)
        if img.shape[0]<self.opt.img_size or img.shape[1]<self.opt.img_size:
            img = np.resize(img, (self.opt.img_size, self.opt.img_size,3))
        ### input label
        label_path = self.label_paths[index]
        # label = np.array(Image.open(label_path), dtype=np.uint8)
        label = np.array(cv2.imread(label_path), dtype=np.uint8)

        # print('label',label.shape)
        if len(label.shape)==3:
            label=label[:,:,0]
        if label.shape[0] < self.opt.img_size or label.shape[1]<self.opt.img_size:
            label = np.resize(label, (self.opt.img_size, self.opt.img_size))
        angleFlag=np.random.rand() > 0.5
        weak_params = {
            # 'angle': np.random.uniform(-30, 30),  # 旋转角度
            'angle': random.choice([0,90,180,270]) if not angleFlag else np.random.uniform(-30, 30),  # 旋转角度
            'translate': (np.random.uniform(-10, 10), np.random.uniform(-10, 10)),  # 平移
            'scale': np.random.uniform(0.5, 1) if not angleFlag else np.random.uniform(0.2, 0.5),  # 缩放
            # 'shear': np.random.uniform(-10, 10),  # 剪切
            'shear': 0,  # 剪切
            'flip': np.random.rand() > 0.5  # 镜像翻转，50%的概率
        }
        affine_matrix=self.get_affine_matrix(weak_params)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # imgWeak = cv2.warpAffine(img, affine_matrix, (self.opt.img_size, self.opt.img_size),)
        # imgWeak = cv2.warpAffine(img, affine_matrix, (self.opt.img_size, self.opt.img_size),
        #                                    flags=cv2.INTER_CUBIC,  # 使用三次样条插值
        #                                    borderMode=cv2.BORDER_CONSTANT,  # 添加常数边界
        #                                    borderValue=(250, 250, 250))  # 白色填充
        # # imgWeak = np.asarray(imgWeak.convert('RGB'))
        # labelWeak = cv2.warpAffine(label, affine_matrix, (self.opt.img_size, self.opt.img_size),
        #                          flags=cv2.INTER_NEAREST,  #
        #                          borderMode=cv2.BORDER_CONSTANT,  # 添加常数边界
        #                          borderValue=(250, 250, 250))  # 白色填充

        # print('labelWeak',label.shape,labelWeak.shape)
        # imgWeak,labelWeak=self.transform(img, label, weak_params)
        # image_transformed = TF.affine(img, **params, fillcolor=0)
        # label_transformed = TF.affine(label, **params, fillcolor=0)

        # img = self.transform(img)
        # 注意：对于标签的变换可能需要不同的处理方式，这里仅为示例
        # label = self.transform(label)

        # 应用随机仿射变换
        # img_pil = Image.fromarray(img)  # 转换为PIL图像，以便使用transform
        # label_pil = Image.fromarray(label).convert('L')  # 转换为灰度PIL图像
        # img_transformed, label_transformed, weak_params = self.random_affine(img_pil, label_pil)
        # ... 转换为Tensor的代码 ...
        # img_tensor = T.ToTensor()(imgWeak)
        # label_tensor = T.ToTensor()(labelWeak)
        # img_full = T.ToTensor()(img)
        # label_full = T.ToTensor()(label)
        # img_full,label_full=TF.to_tensor(img), TF.to_tensor(label)
        # img_tensor, label_tensor = TF.to_tensor(imgWeak), TF.to_tensor(labelWeak)
        # imgWeak, labelWeak=self.transformcon(imgWeak,labelWeak)
        img_full, label_full=self.transformcon(img,label)

        # img_tensor, label_tensor=self.transformcon(imgWeak,labelWeak)

        input_dict = {'img_full': img_full,'label_full': label_full,'img': img_full, 'label': label_full,
                      'imgpath': imgpath, 'label_path': label_path,'weak_params':affine_matrix}
        return input_dict


    def __len__(self):
        return len(self.label_paths) // self.opt.batch_size * self.opt.batch_size
    def transformcon(self, img, lbl):
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


        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()


        return img, lbl


