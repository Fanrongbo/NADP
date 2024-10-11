import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.global_avg_pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPPModule, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):
    """
    Simplified DeepLabV3 for image segmentation.
    """
    def __init__(self, num_classes, atrous_rates=[6, 12, 18],args=None):
        super(DeepLabV3, self).__init__()

        # self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = ResNet(args)
        self.aspp = ASPPModule(2048,atrous_rates, 256)
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
        # print(x4.shape)
        # x = self.aspp(x4)
        # # print('assp',x.shape,x3.shape)
        # x = self.decoder(x, x3)
        # print(x4.shape)#([14, 2048, 16, 16])
        assp = self.aspp(x4)
        outfeat = self.classifier(assp)
        out = F.interpolate(outfeat, size=(h, w), mode='bilinear', align_corners=True)
        return out


# # Example usage
# model = DeepLabV3(in_channels=3, num_classes=21)
# print(model)
