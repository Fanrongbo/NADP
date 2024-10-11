import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import VOCSegmentation
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
import numpy as np
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


class DeepLabV3PlusMultiPrototypeSingleKeyDrop(nn.Module):
    def __init__(self, num_classes=21,args=None,n_cluster=1,ratio=0.1):
        super(DeepLabV3PlusMultiPrototypeSingleKeyDrop, self).__init__()
        self.n_cluster=n_cluster
        self.prototypeN = 1
        self.num_classes=num_classes
        self.featDim = 128
        self.size=32
        self.ratio=ratio

        # self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        print('num_classesnum_classesnum_classes',num_classes)
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
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, self.featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),)
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        # nn.ConvTranspose2d(64, self.featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.ReLU()
        #     )
        self.classifier = nn.Conv2d(self.featDim, num_classes, 1)
        self.key = nn.Conv2d(self.featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(self.featDim * self.prototypeN, 128, kernel_size=1))

        # self.one_key_attention = OneKeyAttation(self.featDim, self.num_classes, self.prototypeN, self.size, self.ratio, self.n_cluster)

    def forward(self, x,DomainLabel=0,maskParam=None,ProtoInput=None,GetProto=False,DomainTrain=False):
        # for j in range(self.n_cluster):
        #     setattr(self,f'ProtoCurrentOut_{j}', [])
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)#[10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        with torch.no_grad():
            pseudo_out = self.classifier(assp_features)#([10, 6, 16, 16])
        #     # mask = torch.argmax(output_feature.detach(), dim=1).unsqueeze(1)
            mask = torch.argmax(pseudo_out, dim=1).unsqueeze(1)
        prototypes = []
        ProtoCurrentOut = []
        if DomainLabel == 0 or GetProto:
            # 计算每个类别的原型
            for i in range(self.num_classes):
                class_mask = (mask == i).float()
                if maskParam is not None:
                    class_mask = class_mask * maskParam  # Apply maskParam
                if class_mask.sum() > 0:  # 确保类别在批次中存在
                    # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                    prototypeC = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]) + 1e-5)#([10, 256])
                    if GetProto:
                        # getattr(self, f'ProtoCurrentOut_{pp}').append(prototypeC.unsqueeze(1).unsqueeze(1))
                        ProtoCurrentOut.append(prototypeC.unsqueeze(1))
                    # for pp in range(self.n_cluster):
                    if DomainLabel == 0:
                        prototypes.append(prototypeC.unsqueeze(1))
                else:
                    if GetProto:
                        # getattr(self, f'ProtoCurrentOut_{pp}').append(zero_prototype.unsqueeze(1).unsqueeze(1))
                        # ProtoCurrentOut.append(zero_prototype.unsqueeze(1))
                        ProtoCurrentOut.append(ProtoInput[i][:,0,:].unsqueeze(1))
                    # for pp in range(self.n_cluster):
                    if DomainLabel == 0:
                        prototypes.append(zero_prototype.unsqueeze(1))
                            # ProtoCurrentOut.append(zero_prototype.unsqueeze(1))
            if prototypes != []:
                prototypes = torch.cat([p.unsqueeze(1) for p in prototypes], dim=1)  # ([5, 12, 1, 128])
                prototypes = prototypes.repeat(1, 1, 2, 1)#([5, 6, 2, 128])

        if DomainLabel==1 and ProtoInput!=None:
            prototypes=ProtoInput
        # assp_weighted, prototypesOut=self.one_key_attention(prototypes,assp_features,DomainTrain)




        if DomainTrain:
            prototypes=prototypes.view(prototypes.size(0),12,prototypes.size(-1))
            # 在维度2和3上随机选择位置
            random_indices = np.random.choice(self.size * self.size,
                                              int(self.size * self.size * self.ratio),
                                              replace=False)  # 随机选择位置
            x_coords = random_indices // self.size  # 计算随机选择的x坐标
            y_coords = random_indices % self.size  # 计算随机选择的y坐标
            # 对于每个随机选择的位置
            for i in range(assp_features.size(0)):
                for x, y in zip(x_coords, y_coords):
                    # 获取深层特征的当前特征
                    feature = assp_features[i, :, x, y]  # 特征向量
                    # 计算与原型特征之间的L2距离
                    distances = torch.norm(prototypes[i] - feature.unsqueeze(0), dim=1)
                    # 找到距离最小的原型特征的索引
                    closest_prototype_idx = torch.argmin(distances)
                    # 用最接近的原型特征替换当前位置的深层特征
                    assp_features[i, :, x, y] = prototypes[i, closest_prototype_idx]
            # print('assp_features',assp_features.shape,prototypes.shape,prototypes.shape)#torch.Size([10, 128, 32, 32]) torch.Size([10, 6, 2, 128])
            prototypes=prototypes.view(prototypes.size(0),6,2,prototypes.size(-1))
        key_output = self.key(assp_features)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        prototypesOut = []
        similarityList = []
        for i in range(prototypes.size(1)):
            for j in range(prototypes.size(2)):
                query_output = getattr(self, f'query_{i}')(
                    prototypes[:, i, j, :].unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))
                similarity = torch.bmm(query_output,
                                       key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat / similarityCat.size(1) * self.n_cluster, dim=1)
        similarityWeiht = similarityWeiht.mean(dim=1)
        # similarityWeiht,_=torch.max(similarityWeiht,dim=1)
        assp_weighted = assp_features * similarityWeiht.unsqueeze(1)








        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        # print(x.shape)
        xup = nn.functional.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=True)
        # assp = nn.functional.interpolate(assp_features, size=(h//4, w//4), mode='bilinear', align_corners=True)
        concat_prototypes = torch.cat([p.unsqueeze(1) for p in prototypesOut], dim=1)#[10, 6, 256])
        # concat_ProtoCurrentOut=[]
        if GetProto:
            # for pp in range(self.n_cluster):
            #     concat_ProtoCurrentOut.append(torch.cat([p for p in getattr(self, f'ProtoCurrentOut_{pp}')], dim=1))#[10, 6, 256])
            # concat_ProtoCurrentOut=torch.cat([p for p in concat_ProtoCurrentOut], dim=2)
            concat_ProtoCurrentOut=torch.cat([p for p in ProtoCurrentOut], dim=1)
        else:
            concat_ProtoCurrentOut=None
        # print('concatenated_prototypes',concatenated_prototypes.shape)
        return xup,[concat_prototypes,concat_ProtoCurrentOut],[assp_features,assp_weighted,x],[prototypes,assp_features,DomainTrain]

class OneKeyAttation(nn.Module):
    def __init__(self,featDim,num_classes,prototypeN,size,ratio,n_cluster):
        super(OneKeyAttation, self).__init__()

        self.size=size
        self.ratio=ratio
        self.n_cluster=n_cluster

        self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        for i in range(num_classes):
            setattr(self, f'query_{i}', nn.Conv2d(featDim * prototypeN, 128, kernel_size=1))
    def forward(self, prototypes,assp_features,DomainTrain):
        if DomainTrain:
            prototypes=prototypes.view(prototypes.size(0),12,prototypes.size(-1))
            # 在维度2和3上随机选择位置
            random_indices = np.random.choice(self.size * self.size,
                                              int(self.size * self.size * self.ratio),
                                              replace=False)  # 随机选择位置
            x_coords = random_indices // self.size  # 计算随机选择的x坐标
            y_coords = random_indices % self.size  # 计算随机选择的y坐标
            # 对于每个随机选择的位置
            for i in range(assp_features.size(0)):
                for x, y in zip(x_coords, y_coords):
                    # 获取深层特征的当前特征
                    feature = assp_features[i, :, x, y]  # 特征向量
                    # 计算与原型特征之间的L2距离
                    distances = torch.norm(prototypes[i] - feature.unsqueeze(0), dim=1)
                    # 找到距离最小的原型特征的索引
                    closest_prototype_idx = torch.argmin(distances)
                    # 用最接近的原型特征替换当前位置的深层特征
                    assp_features[i, :, x, y] = prototypes[i, closest_prototype_idx]
            # print('assp_features',assp_features.shape,prototypes.shape,prototypes.shape)#torch.Size([10, 128, 32, 32]) torch.Size([10, 6, 2, 128])
            prototypes=prototypes.view(prototypes.size(0),6,2,prototypes.size(-1))
        key_output = self.key(assp_features)
        key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # print('key_output',key_output.shape)
        prototypesOut = []
        similarityList = []
        for i in range(prototypes.size(1)):
            for j in range(prototypes.size(2)):
                query_output = getattr(self, f'query_{i}')(
                    prototypes[:, i, j, :].unsqueeze(-1).unsqueeze(-1))
                query_output = query_output.view(query_output.size(0), -1,
                                                 query_output.size(1))  # Reshape to [10, 1, 128]
                prototypesOut.append(query_output.squeeze(1))
                similarity = torch.bmm(query_output,
                                       key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
        similarityCat = torch.cat(similarityList, dim=1)
        similarityWeiht = F.softmax(similarityCat / similarityCat.size(1) * self.n_cluster, dim=1)
        similarityWeiht = similarityWeiht.mean(dim=1)
        # similarityWeiht,_=torch.max(similarityWeiht,dim=1)
        assp_weighted = assp_features * similarityWeiht.unsqueeze(1)
        return assp_weighted,prototypesOut
