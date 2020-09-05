# Zhaolun Zou 08/14/2019
import torch.nn as nn
from models.unet_parts import *

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        return


class Deconv_Decoder(Decoder):
    def __init__(self, class_num=13):
        super(Deconv_Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2,
                                          output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2,
                                          output_padding=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=2,
                                          output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2,
                                          output_padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2,
                                          output_padding=1)
        self.conv3 = nn.Conv2d(32, class_num, kernel_size=1)
        return

    def forward(self, x):
        x1 = self.deconv1(x)
        x = self.deconv2(x1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.deconv5(x)
        # x = self.conv3(x)
        # return x # regular decoder
        return x1 # fusion at middle Unet

class Unet_Decoder_18(Decoder):
    def __init__(self, class_num=13):
        super(Unet_Decoder_18, self).__init__()
        self.up1 = res_up(768, 256)
        self.up2 = res_up(384, 128)
        self.up3 = res_up(192, 64)
        self.up4 = res_up(128, 64)
        self.deconv = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv = nn.Conv2d(64, 3, kernel_size=1)
        return

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.deconv(x)
        x = self.conv(x)
        return x # regular decoder
        # return x1 # fusion at middle Unet

class Unet_Decoder_50(Decoder):
    def __init__(self, class_num=13):
        super(Unet_Decoder_50, self).__init__()
        self.up1 = res_up(3072, 512)
        self.up2 = res_up(1024, 256)
        self.up3 = res_up(512, 128)
        self.up4 = res_up(192, 64)
        self.deconv = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.conv = nn.Conv2d(64, 3, kernel_size=1)
        return

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.deconv(x)
        x = self.conv(x)
        return x # regular decoder
        # return x1 # fusion at middle Unet