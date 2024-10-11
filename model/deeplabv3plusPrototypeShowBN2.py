import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module with Domain-specific Batch Normalization
    """

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        self.convs = nn.ModuleList()
        for rate in atrous_rates:
            self.convs.append(nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                'bn_source': nn.BatchNorm2d(out_channels),
                'bn_target': nn.BatchNorm2d(out_channels),
                'relu': nn.ReLU(inplace=True)
            }))

        self.pool = nn.ModuleDict({
            'pool': nn.AdaptiveAvgPool2d(1),
            'conv': nn.Conv2d(in_channels, out_channels, 1, bias=False),
            'bn_source': nn.BatchNorm2d(out_channels),
            'bn_target': nn.BatchNorm2d(out_channels),
            'relu': nn.ReLU(inplace=True)
        })
        self.project = nn.ModuleDict({
            'conv': nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            'bn_source': nn.BatchNorm2d(out_channels),
            'bn_target': nn.BatchNorm2d(out_channels),
            'relu': nn.ReLU(inplace=True)
        })

    def forward(self, x, domain_label):
        res = []
        for conv in self.convs:
            if domain_label == 0:
                res.append(conv['relu'](conv['bn_source'](conv['conv'](x))))
            else:
                res.append(conv['relu'](conv['bn_target'](conv['conv'](x))))

        if domain_label == 0:
            pool_out = self.pool['relu'](self.pool['bn_source'](self.pool['conv'](self.pool['pool'](x))))
        else:
            pool_out = self.pool['relu'](self.pool['bn_target'](self.pool['conv'](self.pool['pool'](x))))

        res.append(F.interpolate(pool_out, size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)

        if domain_label == 0:
            return self.project['relu'](self.project['bn_source'](self.project['conv'](res)))
        else:
            return self.project['relu'](self.project['bn_target'](self.project['conv'](res)))


class DeepLabV3PlusSimGlobalLinearKLBN(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLBN, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_classes = num_classes
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.asppconv = nn.Sequential(
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(), )

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        for i in range(num_classes):
            setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
            setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
        # self.normP = nn.LayerNorm(128)

        # Domain-specific BN layers for initial ResNet layers
        # self.bn1_source = nn.BatchNorm2d(64)
        # self.bn1_target = nn.BatchNorm2d(64)

    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None, getPFlag=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        assp_features = self.aspp(x4, DomainLabel)
        assp_features=self.asppconv(assp_features)
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        prototypes = []
        if ProtoInput is None:
            with torch.no_grad():
                pseudo_out = self.classifier(assp_features)
                mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
                for i in range(self.num_classes):
                    class_mask = (mask == i).float()
                    if maskParam is not None:
                        class_mask = class_mask * maskParam
                    if class_mask.sum() > 0:
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(-1)
                prototypes = prototypesC
        else:
            prototypes = ProtoInput

        query_outputList = []
        similarityList = []
        key_output = assp_features

        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                deep_features_normalized = F.normalize(key_output, p=2, dim=1)
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)
                similarity = (deep_features_normalized * proto_features_normalized).sum(dim=1).unsqueeze(1)
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2), assp_features.size(3))
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1, prototype.size(1))
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat, dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        assp_weighted = assp_features * similarityWeihtMax.unsqueeze(1)
        x = self.classifier(assp_features)
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}
