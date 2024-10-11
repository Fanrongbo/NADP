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
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21,args=None):
        super(DeepLabV3Plus, self).__init__()
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

        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        self.classifier = nn.Conv2d(256, num_classes, 1)

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

        assp = self.aspp(x4)
        # print('assp',x.shape,x3.shape)
        x = self.classifier(assp)
        #
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        assp = nn.functional.interpolate(assp, size=(h//4, w//4), mode='bilinear', align_corners=True)

        # print('x',x.shape)

        return xup,assp


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module
    """

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.convs = nn.ModuleList(modules)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusRecon(nn.Module):
    def __init__(self, args=None):
        super(DeepLabV3PlusRecon, self).__init__()
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

        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        # self.decoder = Decoder(num_classes=num_classes)
        # self.classifier = nn.Conv2d(256, num_classes, 1)

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

        # assp = self.aspp(x4)
        # print('assp',x.shape,x3.shape)
        # x = self.classifier(assp)
        #
        # xup = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x4
