import torch
import torch.nn as nn
import torchvision.models as models


class UNetMobileNetV3(nn.Module):
    def __init__(self, num_classes=4):  # Исправлено на 4 класса
        super(UNetMobileNetV3, self).__init__()

        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = mobilenet_v3.features

        self.decoder1 = self._conv_block(576, 256)
        self.decoder2 = self._conv_block(256, 128)
        self.decoder3 = self._conv_block(128, 64)
        self.decoder4 = self._conv_block(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.encoder[0:3](x)
        x2 = self.encoder[3:6](x1)
        x3 = self.encoder[6:13](x2)
        x4 = self.encoder[13:](x3)

        x = self.decoder1(x4)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        return self.final_conv(x)


def create_model(num_classes=4):
    return UNetMobileNetV3(num_classes=num_classes)
