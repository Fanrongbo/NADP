import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import VOCSegmentation
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
# 定义DeepLab v3模型
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3, self).__init__()
        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

        if args.mode == 'imp':
            pretrained = '../pretrain_model/resnet50-19c8e357.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_120':
            pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch120/millionAID_224_None/0.0005_0.05_192/resnet/100/ckpt.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'rsp_300':
            self.resnet = ResNet(args)
            # pretrained = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
            # pretrained_weights = torch.load(pretrained)
            # self.resnet.load_state_dict(pretrained_weights)
            # pretrained = '../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth'
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.mode=='nopre':
            self.resnet = models.resnet50(weights=None)
        # self.resnet = models.resnet50(weights=False)
        # # 加载你的本地权重文件
        # weight_path = '/data/project_frb/SegDA/Segonly/model/RSP/pretrain/rsp-resnet-50-ckpt.pth'
        # state_dict = torch.load(weight_path)
        # # 更新模型的状态字典
        # self.resnet.load_state_dict(state_dict)

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x = self.aspp(x4)
        # print('assp',x.shape,x3.shape)
        x = self.decoder(x, x3)

        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

# 定义ASPP模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        # print('in_channels * 5',in_channels * 5,out_channels)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        [b, c, row, col] = x.size()
        x1 = self.bn1(self.conv1x1_1(x))
        x2 = self.bn2(self.conv3x3_1(x))
        x3 = self.bn3(self.conv3x3_2(x))
        x4 = self.bn4(self.conv3x3_3(x))
        x5 = self.bn5(self.conv1x1_2(self.avg_pool(x)))
        x5 = F.interpolate(x5, (row, col), None, 'bilinear', True)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print(x.shape)
        x = self.conv_cat(x)

        return x

# 定义Decoder模块
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1024, 48, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(256+48, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, x_low):
        x = nn.functional.interpolate(x, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.bn1(self.conv1(x_low))
        x = torch.cat((x, x_low), dim=1)
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.conv4(x)

        return x