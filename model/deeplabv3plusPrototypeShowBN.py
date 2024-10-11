import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DomainAdaptationBatchNorm(nn.Module):
    def __init__(self, num_features, num_domains=2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DomainAdaptationBatchNorm, self).__init__()
        self.num_domains = num_domains
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)
        ])
        self.current_domain = 0

    def set_domain(self, domain_index):
        if domain_index >= self.num_domains or domain_index < 0:
            raise ValueError("Invalid domain index")
        self.current_domain = domain_index

    def forward(self, x):
        return self.bn_layers[self.current_domain](x)

    def load_pretrained_params(self, pretrained_bn):
        for bn in self.bn_layers:
            bn.load_state_dict(pretrained_bn.state_dict())


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18], num_domains=2):
        super(ASPPModule, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                DomainAdaptationBatchNorm(out_channels, num_domains=num_domains),
                nn.ReLU(inplace=False)
            ))

        self.convs = nn.ModuleList(modules)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            DomainAdaptationBatchNorm(out_channels, num_domains=num_domains),
            nn.ReLU(inplace=False)
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * out_channels + out_channels, out_channels, 1, bias=False),
            DomainAdaptationBatchNorm(out_channels, num_domains=num_domains),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(F.interpolate(self.pool(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusSimGlobalLinearKLBN(nn.Module):
    def __init__(self, num_classes=21, args=None, n_cluster=1, num_domains=2):
        super(DeepLabV3PlusSimGlobalLinearKLBN, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Collect all BatchNorm layers to replace them after iteration
        bn_layers = []
        for name, module in self.resnet.named_modules():

            if isinstance(module, nn.BatchNorm2d) and 'downsample' not in name:
                # print('name',name)
                bn_layers.append((name, module))

        # Replace BatchNorm layers in ResNet with DomainAdaptationBatchNorm and load pretrained parameters
        for name, module in bn_layers:
            num_features = module.num_features
            modules = name.split('.')
            mod = self.resnet
            for mod_name in modules[:-1]:
                # print('mod_name',mod_name)
                mod = mod._modules[mod_name]
            new_bn = DomainAdaptationBatchNorm(num_features, num_domains=num_domains)
            new_bn.load_pretrained_params(module)
            mod._modules[modules[-1]] = new_bn

        self.aspp = nn.Sequential(
            ASPPModule(in_channels=2048, out_channels=256, num_domains=num_domains),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.n_cluster = n_cluster
        self.prototypeN = 1

        # for i in range(num_classes):
        #     setattr(self, f'a_{i}', nn.Parameter(torch.ones(1), requires_grad=True))
        #     setattr(self, f'b_{i}', nn.Parameter(torch.zeros(1), requires_grad=True))

        # self.normP = nn.LayerNorm(128)

    def set_domain(self, domain_index):
        for name, module in self.named_modules():
            if isinstance(module, DomainAdaptationBatchNorm):
                # print('name',domain_index,name,module)
                module.set_domain(domain_index)

    def forward(self, x, DomainLabel=0, maskParam=None, ProtoInput=None, getPFlag=False):
        self.set_domain(DomainLabel)
        h, w = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        assp_features = self.aspp(x4)
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



# 测试
# if __name__ == "__main__":
#     model = DeepLabV3PlusSimGlobalLinearKLBN(num_classes=21, num_domains=2)
#     input_tensor = torch.randn(1, 3, 256, 256)
#     output = model(input_tensor, DomainLabel=0)
#     print(output)
