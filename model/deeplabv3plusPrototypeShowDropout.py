
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import VOCSegmentation
import torchvision.models as models
import torch.nn.functional as F
from model.RSP.resnet import ResNet
from util.ProtoDistValidate import *
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

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
                nn.ReLU(inplace=False)
            ))

        self.convs = nn.ModuleList(modules)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)
class DeepLabV3PlusSimGlobalLinearKLDrop(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1):
        super(DeepLabV3PlusSimGlobalLinearKLDrop, self).__init__()
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
        elif args.mode == 'seco':
            pretrained = '../pretrain_model/seco_resnet50_1m.pth'
            pretrained_weights = torch.load(pretrained)
            self.resnet.load_state_dict(pretrained_weights)
        elif args.mode == 'office':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        elif args.mode == 'nopre':
            self.resnet = models.resnet50(weights=None)
        self.num_classes = num_classes
        # self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        featDim = 128
        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256),
            nn.ConvTranspose2d(256, featDim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU())

        self.classifier = nn.Conv2d(featDim, num_classes, 1)

        self.n_cluster = n_cluster
        self.prototypeN = 1
        # self.key = nn.Conv2d(featDim, featDim, kernel_size=3, padding=1)
        # for i in range(num_classes):
        #     # setattr(self, f'query_{i}', nn.Conv2d(featDim * self.prototypeN, 128, kernel_size=1))
        #     setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
        #     setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))
            # setattr(self, f'key_{i}', nn.Conv2d(256, 128, kernel_size=1))
        self.normP=nn.LayerNorm(128)
        self.one_key_attention = SingleKeyAttention(featDim=featDim, num_classes=self.num_classes,
                                                    n_cluster=self.n_cluster,size=32,ratio=0.1)
    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None,getPFlag=False,Drop=False):
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)  # [10, 256, 16, 16])
        zero_prototype = torch.zeros(assp_features.size(0), assp_features.size(1), dtype=assp_features.dtype,
                                     device=assp_features.device)
        if getPFlag:
            return assp_features
        # x = self.classifier(assp_features)  # ([10, 6, 16, 16])
        prototypes = []
        DomainTrain=False
        if ProtoInput == None:
            DomainTrain=False
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
                        # prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (class_mask.sum(dim=[2, 3]+1))
                        prototype = (assp_features * class_mask).sum(dim=[2, 3]) / (
                                    class_mask.sum(dim=[2, 3]) + 1e-5)  # ([10, 256])
                        prototypes.append(prototype.unsqueeze(1))
                    else:
                        # for pp in range(self.n_cluster):
                        prototypes.append(zero_prototype.unsqueeze(1))
                prototypesC = torch.cat(prototypes, dim=1).unsqueeze(-1).unsqueeze( -1)  # prototypes torch.Size([10, 6, 128, 1, 1])
                prototypes = prototypesC
        elif ProtoInput != None:
            prototypes = ProtoInput


        if Drop:
            assp_features=self.one_key_attention(assp_features,prototypes)

        query_outputList = []
        # prototypesOut = []
        similarityList = []
        # key_output = self.key(assp_features)
        key_output=assp_features
        for ii in range(prototypes.size(1)):
            prototype = prototypes[:, ii]
            if prototype is not None:

                deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
                proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
                similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]

                similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]

                similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
                                             assp_features.size(3))  # ([10, 1, 16, 16])
                similarityList.append(similarity)
                query_output = prototype.view(prototype.size(0), -1,
                                                 prototype.size(1))  # Reshape to [10, 1, 128]
                query_outputList.append(query_output)

        query_outputcat = torch.cat(query_outputList, dim=1).unsqueeze(-1).unsqueeze(-1)
        # print('query_outputcat',query_outputcat.shape,prototypes.shape,GlobalProto_transOut.shape)
        similarityCat = torch.cat(similarityList, dim=1)#([10, 30, 32, 32])
        similarityWeiht = F.softmax(similarityCat, dim=1)#torch.Size([10, 30, 32, 32])
        # similarityWeihtMean=similarityWeiht.mean(dim=1)
        similarityWeihtMax, _ = torch.max(similarityWeiht, dim=1)
        # print('assp_features',assp_features.shape,similarityWeihtMax.shape,similarityCat.shape)
        # assp_weighted = (assp_features + similarityWeihtMax.unsqueeze(1))/2
        assp_weighted = assp_features*(1+similarityWeihtMax.unsqueeze(1))

        x = self.classifier(assp_weighted)  # ([10, 6, 16, 16])
        xup = nn.functional.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        return {'out': x, 'outUp': xup}, {'CurrentPorotype': None, 'GetProto': prototypes, 'query': query_outputcat}, \
               {'asspF': assp_features, 'asspFW': assp_weighted, 'cat': similarityCat,
                'Weight': [similarityWeiht, similarityWeihtMax]}

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.resnet.conv1)
        b.append(self.resnet.bn1)
        b.append(self.resnet.layer1)
        b.append(self.resnet.layer2)
        b.append(self.resnet.layer3)
        b.append(self.resnet.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.aspp.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self):
        learning_rate=2.5e-4
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]
class SingleKeyAttention(nn.Module):
    def __init__(self, featDim, num_classes,n_cluster,size,ratio):
        super(SingleKeyAttention, self).__init__()
        self.n_cluster=n_cluster
        self.size=size
        self.ratio=ratio
        # self.key = nn.Conv2d(featDim, 128, kernel_size=1)
        # for i in range(num_classes):
        #     setattr(self, f'query_{i}', nn.Conv2d(featDim * 1, 128, kernel_size=1))
    def forward(self, assp_features, prototypes):
        # print('before', assp_features.shape, prototypes.shape)  # torch.Size([10, 128, 32, 32]) torch.Size([10, 6, 2, 128])
        assp_features=assp_features.clone().detach().to(assp_features.device)
        # prototypes=torch.tensor(prototypes,requires_grad=False)
        prototypes = prototypes.view(prototypes.size(0), 18, prototypes.size(2))
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
        # prototypes = prototypes.view(prototypes.size(0), 6, self.n_cluster, prototypes.size(-1))
        # assp_features
        return assp_features
        # # key_output = self.key(assp_features)
        # # key_output = key_output.view(key_output.size(0), key_output.size(1), -1)  # Reshape to [10, 128, 16*16]
        # key_output=assp_features
        #
        # # prototypesOut = []
        # similarityList = []
        # query_outputList=[]
        # for ii in range(prototypes.size(1)):
        #     prototype = prototypes[:, ii]
        #     if prototype is not None:
        #
        #         deep_features_normalized = F.normalize(key_output, p=2, dim=1)  # [B, 128, H, W]
        #         proto_features_normalized = F.normalize(prototype, p=2, dim=1)  # [B, 128, 1, 1]
        #         similarity = (deep_features_normalized * proto_features_normalized) # [B, H, W]
        #
        #         similarity = similarity.sum(dim=1).unsqueeze(1) # [B, H, W]
        #
        #         similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
        #                                      assp_features.size(3))  # ([10, 1, 16, 16])
        #         similarityList.append(similarity)
        #         query_output = prototype.view(prototype.size(0), -1,
        #                                          prototype.size(1))  # Reshape to [10, 1, 128]
        #         query_outputList.append(query_output)
        # # for i in range(prototypes.size(1)):
        # #     for j in range(prototypes.size(2)):
        # #
        # #         query_output = getattr(self, f'query_{i}')(
        # #             prototypes[:, i, j, :].unsqueeze(-1).unsqueeze(-1))
        # #         query_output = query_output.view(query_output.size(0), -1,
        # #                                          query_output.size(1))  # Reshape to [10, 1, 128]
        # #         prototypesOut.append(query_output.squeeze(1))
        # #         similarity = torch.bmm(query_output,
        # #                                key_output)  # Perform batch matrix multiplication#torch.Size([10, 1, 256])
        # #         similarity = similarity.view(assp_features.size(0), 1, assp_features.size(2),
        # #                                      assp_features.size(3))  # ([10, 1, 16, 16])
        # #         # print('key_output',key_output.shape,query_output.shape,similarity.shape)
        # #         similarityList.append(similarity)
        # similarityCat = torch.cat(similarityList, dim=1)
        # similarityWeiht = F.softmax(similarityCat / similarityCat.size(1) * prototypes.size(2), dim=1)
        #
        # # similarityWeiht = F.softmax(similarityCat, dim=1)
        # similarityWeiht, _ = torch.max(similarityWeiht, dim=1)
        # # similarityWeiht = similarityWeiht.mean(dim=1)
        #
        # assp_weighted = assp_features * similarityWeiht.unsqueeze(1)
        # return assp_weighted,query_outputList