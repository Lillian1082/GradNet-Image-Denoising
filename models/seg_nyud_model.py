# Zhaolun Zou 08/14/2019

import torch.nn as nn
import models.encoders as encoders
import models.decoders as decoders


class Seg_Model(nn.Module):
    def __init__(self):
        super(Seg_Model, self).__init__()
        self.encoder = encoders.Resnet18_Encoder(pretrained=True)
        # self.encoder = encoders.Resnet50_Encoder(pretrained=True)
        # self.decoder = decoders.Deconv_Decoder()
        self.decoder = decoders.Unet_Decoder_18()
        return

    def forward(self, x):
        x5, x4, x3, x2, x1 = self.encoder(x)
        x = self.decoder(x5, x4, x3, x2, x1)
        return x

    def unet_parameters(self):
        return self.unet.parameters()

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()
