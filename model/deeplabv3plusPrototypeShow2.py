import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DeepLabV3PlusSimGlobalLinearKL(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKL, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))

        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.classifier= nn.Conv2d(128, num_classes, kernel_size=1)
        self.num_classes=num_classes
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False):
        h, w = x.size()[2:]

        size = x.shape[2:]
        x1 = self.backbone(x)
        x2 = self.aspp(x1)
        x2 = F.interpolate(x2, size=(size[0] // 4, size[1] // 4), mode='bilinear', align_corners=True)
        low_level_feat = self.backbone[0:4](x)
        low_level_feat = F.interpolate(low_level_feat, size=(size[0] // 4, size[1] // 4), mode='bilinear',
                                       align_corners=True)
        x = torch.cat((x2, low_level_feat), dim=1)
        assp_features = self.decoder(x)
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            # print('assp_features',assp_features.shape)
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)  # ([10, 6, 16, 16])
            #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out.detach(), dim=1).unsqueeze(1)
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                            class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                    prototypes.append(prototype.unsqueeze(1))
                else:
                    prototypes.append(zero_prototype.unsqueeze(1))
            prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze(
                -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
        if ProtoInput == None:
            prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput
        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output = assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:
                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized)  # [B, H, W]

                similarity = similarity.sum(dim=1).unsqueeze(1)  # [B, H, W]

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                              prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)  # ([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)  # torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        # print('assp_features',assp_features.shape,similarityWeihtMax.shape,similarityCat.shape)
        assp_weighted = (assp_features + similarityWeihtMax.unsqueeze(1)) / 2

        x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        # xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': x}, {'CurrentPorotype': prototypesC, 'GetProto': prototypes,
                                          'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}



        # x=self.classifier(x)
        # print('x',x.shape)
        # x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        #
        # return x

#
# # Example usage:
# model = DeepLabV3PlusSimGlobalLinearKL(num_classes=21)
# input_tensor = torch.randn(1, 3, 512, 512)
# output,_,_ = model(input_tensor)
# print(output['outUp'].size())
