import torch.nn as nn

class Denoise_AutoEncoders(nn.Module):
    def __init__(self):
        super(Denoise_AutoEncoders, self).__init__()
        # 定义Encoder编码层
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # [, 64, 96, 96]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # [, 64, 96, 96]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), # [, 64, 96, 96]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # [, 256, 48, 48]
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # [, 256, 24, 24]
            nn.BatchNorm2d(256),
        )


    def forward(self, x):
        encoder = self.Encoder(x)

        return encoder
class Denoise_AutoDcoders(nn.Module):
    def __init__(self):
        super(Denoise_AutoDcoders, self).__init__()
        # 定义Decoder解码层，使用线性插值+卷积修正
        self.Decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),  # [, 256, 48, 48]
            nn.Conv2d(256, 128, 3, stride=1, padding=1),  # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),  # [, 64, 96, 96]
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # [, 32, 96, 96]
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),  # [, 3, 96, 96]
            nn.Sigmoid()
        )

    def forward(self, x):
        decoder = self.Decoder(x)
        return decoder

class Denoise_AutoDcoders2(nn.Module):
    def __init__(self):
        super(Denoise_AutoDcoders2, self).__init__()
        # 定义Decoder解码层，使用线性插值+卷积修正
        self.Decoder = nn.Sequential(
            # nn.UpsamplingBilinear2d(scale_factor=4),  # [, 256, 48, 48]
            nn.Conv2d(2048, 128, 3, stride=1, padding=1),  # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.UpsamplingBilinear2d(scale_factor=4),  # [, 256, 48, 48]
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # [, 128, 48, 48]
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=4),  # [, 64, 96, 96]
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # [, 32, 96, 96]
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.UpsamplingBilinear2d(scale_factor=2),  # [, 64, 96, 96]
            # nn.Conv2d(32, 16, 3, stride=1, padding=1),  # [, 32, 96, 96]
            # nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),  # [, 3, 96, 96]
            nn.Sigmoid()
        )

    def forward(self, x):
        decoder = self.Decoder(x)
        return decoder
