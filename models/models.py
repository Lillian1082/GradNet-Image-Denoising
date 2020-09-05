import math

import torch
from torch import nn
from models import common
from models import ops
from models.unet_model import *
from models.seg_nyud_model import *
import numpy as np
from models.cbam import *
#from utils import weights_init_kaiming

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(residual) + x) # addition Structure 6-1 elf.bn2
        return out

class ResidualUnit(nn.Module): # 4 residual block
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        self.block1 = ResidualBlock(channels)
        self.block2 = ResidualBlock(channels)
        self.block3 = ResidualBlock(channels)
        self.block4 = ResidualBlock(channels)
        self.conv = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.ca = CALayer(channels) # with attention
        # self.cbam = CBAM(channels) # with cbam attention
        # self.aoa = AoA(channels)  # with aoa attention

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        mid = torch.cat((x, res), dim=1)
        out = self.conv(mid)
        out = self.ca(out) # with attention
        # out = self.cbam(out) # with cbam attention
        # out = self.aoa(out)
        return out

# class ResidualUnit(nn.Module): # 4 residual block Sparse Attention
#     def __init__(self, channels):
#         super(ResidualUnit, self).__init__()
#         self.block1 = ResidualBlock(channels)
#         self.block2 = ResidualBlock(channels)
#         self.block3 = ResidualBlock(channels)
#         self.block4 = ResidualBlock(channels)
#         # self.ca = CALayer(channels) # with attention
#         # self.cbam = CBAM(channels) # with cbam attention
#         self.aoa = AoA(channels) # with sparse attention
# #
#     def forward(self, x):
#         res = self.block1(x)
#         res = self.block2(res)
#         res = self.block3(res)
#         res = self.block4(res)
#         # out = self.ca(res) # with attention
#         # out = self.cbam(res) # with cbam attention
#         out = self.aoa(res)
#         return (x + out)

class ResidualModule(nn.Module):
    def __init__(self, channels):
        super(ResidualModule, self).__init__()
        self.block1 = ResidualUnit(channels)
        self.block2 = ResidualUnit(channels)
        self.block3 = ResidualUnit(channels) # 3 residual units
        self.block4 = ResidualUnit(channels) # 4 residual units

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        res = self.block3(res)
        res = self.block4(res)
        return (x + res)

# reference: original DnCNN
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False))
    
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

# reference: DnCNN with skip connection
class DnCNN_sk(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_sk, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers)
        self.block3 = nn.Conv2d(in_channels=(features + features), out_channels=3, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        x = torch.cat((block1, block2), dim=1)
        out = self.block3(x)
        return out

class DnCNN_sk_conDenoised(nn.Module): # concatenate denoised image into the noisy image
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_sk_conDenoised, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=(channels + channels), out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers)
        self.block3 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, denoised_img):
        x = torch.cat((x, denoised_img), dim=1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        out = self.block3(block2)
        return out

class DnCNN_conGradient_image(nn.Module): # learn image
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_conGradient_image, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=2*channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers)
        self.block3 = nn.Conv2d(in_channels=(features + features), out_channels=3, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x, gradient):
        x = torch.cat((x, gradient), dim=1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        x = torch.cat((block1, block2), dim=1)
        out = self.block3(x)
        return out

class DnCNN_sk_conGradient(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_sk_conGradient, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers)
        self.block3 = nn.Conv2d(in_channels=(features+features+3), out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, gradient):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        x = torch.cat((block1, block2), dim=1)
        x = torch.cat((x, gradient), dim=1)
        out = self.block3(x)
        return out


class DnCNN_sk_conGradient_before(nn.Module): # The version for clear image
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN_sk_conGradient_before, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=(features + 3), out_channels=(features + 3), kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features + 3))
            layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*layers)
        # self.block3 = nn.Sequential(
        #    nn.Conv2d(in_channels=(2*(features + 3)), out_channels=features, kernel_size=kernel_size,
        #                        padding=padding, bias=False),
        #    nn.ReLU(inplace=True)
        # )
        self.block3 = nn.Conv2d(in_channels=(2*(features + 3)), out_channels=features, kernel_size=kernel_size,
                                padding=padding, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.block4 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, gradient):
        block1 = self.block1(x)
        x = torch.cat((block1, gradient), dim=1)
        block2 = self.block2(x)
        block3 = torch.cat((block2, x), dim=1)
        block4 = self.block3(block3)
        # block4 = self.relu(block4)
        # from tqdm import tqdm
        # import torchvision
        # print(block4.shape)
        # bb = block4.cpu()
        # for i in range(block4.shape[1]):
        #     b = bb[0, i, :, :]
        #     torchvision.utils.save_image(b, 'bt_final/{0:d}.png'.format(i))
        # assert 0
        out = self.block4(block4)
        return out

class Resnet_sk(nn.Module): # Resnet with skip connection
    def __init__(self, channels):
        super(Resnet_sk, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualModule(features)
        self.block3 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        out = self.block3(block2)
        return x + out #out #

class Res_conGradient_before(nn.Module):
    def __init__(self, channels):
        super(Res_conGradient_before, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualModule((features+3))
        self.block3 = nn.Conv2d(in_channels=(features + 3), out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, gradient):
        block1 = self.block1(x)
        mid = torch.cat((block1, gradient), dim=1)
        block2 = self.block2(mid)
        out = self.block3(block2)
        return out #(x + out)

class Res_conGradient_onimage(nn.Module):
    def __init__(self, channels):
        super(Res_conGradient_onimage, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=(channels+3), out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualModule(features)
        self.block3 = nn.Conv2d(in_channels= features, out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, gradient):
        mid = torch.cat((x, gradient), dim=1)
        block1 = self.block1(mid)
        block2 = self.block2(block1)
        out = self.block3(block2)
        return (x + out)

class Res_conGradient(nn.Module):
    def __init__(self, channels):
        super(Res_conGradient, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualModule(features)
        self.block3 = nn.Conv2d(in_channels=(features+3), out_channels=3, kernel_size=kernel_size,
                                padding=padding, bias=False)

    def forward(self, x, gradient):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        mid = torch.cat((block2, gradient), dim=1)
        out = self.block3(mid)
        return (x + out)

# class Res_conGradient_before4(nn.Module): # MWavelet cancatenate images
#     def __init__(self, channels):
#         super(Res_conGradient_before4, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.block2 = ResidualModule((features+3))
#         self.block3 = nn.Conv2d(in_channels=(features + 3), out_channels=3, kernel_size=kernel_size,
#                              padding=padding, bias=False)
#
#     def forward(self, x, noisy, gradient):
#         block1 = self.block1(x)
#         mid = torch.cat((block1, gradient), dim=1)
#         block2 = self.block2(mid)
#         out = self.block3(block2)
#         return (noisy + out)
#
# class Res_conGradient_before5(nn.Module): # MWavelet cancatenate features
#     def __init__(self, channels):
#         super(Res_conGradient_before5, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.block2 = ResidualModule((features + 15))
#         self.block3 = nn.Conv2d(in_channels=(features + 15), out_channels=3, kernel_size=kernel_size,
#                                 padding=padding, bias=False)
#
#     def forward(self, x, noisy, gradient):
#         block1 = self.block1(noisy)
#         mid = torch.cat((block1, gradient), dim=1)
#         mid = torch.cat((mid, x), dim=1)
#         block2 = self.block2(mid)
#         out = self.block3(block2)
#         return (noisy + out)
#
# class Res_conGradient_before6(nn.Module): # Multi-scale Denoising
#     def __init__(self, channels):
#         super(Res_conGradient_before6, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         self.downscale = nn.MaxPool2d(2, stride=2)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.block2 = ResidualModule(features)
#         self.block3 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size,
#                                 padding=padding, bias=False)
#         self.upscale1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1)
#         self.upscale2 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1)
#         self.upscale3 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1)
#
#     def forward(self, x): # x (160, 160)
#         scale1_x = self.downscale(x) # x_scale1 (80, 80)
#         scale2_x = self.downscale(scale1_x) # x_scale2 (40, 40)
#
#         x_block1 = self.block1(x)
#         x_block2 = self.block2(x_block1)
#         x_out = self.block3(x_block2)
#
#         scale1_x_block1 = self.block1(scale1_x)
#         scale1_x_block2 = self.block2(scale1_x_block1)
#         scale1_x_out = self.block3(scale1_x_block2)
#         scale1_x_out = self.upscale1(scale1_x_out, output_size=x_out.size())
#
#         scale2_x_block1 = self.block1(scale2_x)
#         scale2_x_block2 = self.block2(scale2_x_block1)
#         scale2_x_out = self.block3(scale2_x_block2)
#         scale2_x_out = self.upscale2(scale2_x_out, output_size=scale1_x.size())
#         scale2_x_out = self.upscale3(scale2_x_out, output_size=x_out.size())
#
#         return (x_out + scale1_x_out + scale2_x_out)/3
#
#
# # reference: Parallel DnCNN with skip connection
# class skDnCNN_unit(nn.Module):
#     def __init__(self, channels, num_of_layers=17):
#         super(skDnCNN_unit, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         layers = []
#         for _ in range(num_of_layers - 2):
#             layers.append(
#                 nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
#                           bias=False))
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
#         self.block1 = nn.Sequential(*layers)
#         self.block2 = nn.Conv2d(in_channels=(features + features), out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = torch.cat((x, block1), dim=1)
#         out = self.block2(block2)
#         return out
#
# class Parallel_skDnCNN(nn.Module):
#     def __init__(self, channels):
#         super(Parallel_skDnCNN, self).__init__()
#         kernel_size = 3
#         padding = 1
#         features = 64
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.block2 = skDnCNN_unit(channels = features)
#         self.block3 = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.block4 = skDnCNN_unit(channels = features)
#         self.block5 = nn.Conv2d(in_channels=(features + features), out_channels=128, kernel_size=kernel_size, padding=padding, bias=False)
#         self.block6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size,
#                                 padding=padding, bias=False)
#         self.block7 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel_size,
#                                 padding=padding, bias=False)
#
#     def forward(self, x1, x2):
#         block1 = self.block1(x1)
#         block2 = self.block2(block1)
#         x1 = torch.cat((block1, block2), dim=1)
#         block3 = self.block3(x2)
#         block4 = self.block4(block3)
#         x2 = torch.cat((block3, block4), dim=1)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.block5(x)
#         x = self.block6(x)
#         out = self.block7(x)
#         return out

def img_gradient_total(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G

def img_gradient(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    a = torch.from_numpy(a).float().unsqueeze(0)
    a = torch.stack((a, a, a))
    conv1.weight = nn.Parameter(a, requires_grad=False)
    conv1 = conv1.cuda()
    G_x = conv1(img)

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    b = torch.from_numpy(b).float().unsqueeze(0)
    b = torch.stack((b, b, b))
    conv2.weight = nn.Parameter(b, requires_grad=False)
    conv2 = conv2.cuda()
    G_y = conv2(img)

    return G_x, G_y

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


def init_weights(modules):
    pass

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class AoA(nn.Module): # Attention on attention
    def __init__(self, channel, reduction=16):
        super(AoA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = CALayer(channel)
        self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        attonatt=self.ca(y2)
        return x * attonatt

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.Merge_Run_dual(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels, out_channels)
        self.r3 = ops.EResidualBlock(in_channels, out_channels)
        # self.g = ops.BasicBlock(in_channels, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        # g = self.g(r3)
        out = self.ca(r3)

        return out


class RIDNET(nn.Module):
    def __init__(self, n_feats, rgb_range):
        super(RIDNET, self).__init__()

        kernel_size = 3

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = ops.BasicBlock(3, n_feats, kernel_size, 1, 1)
        self.conv = nn.Conv2d((3 + n_feats), n_feats, kernel_size, 1, 1)
        self.b1 = Block(n_feats, n_feats)
        self.b2 = Block(n_feats, n_feats)
        self.b3 = Block(n_feats, n_feats)
        self.b4 = Block(n_feats, n_feats)

        self.tail = nn.Conv2d(n_feats, 3, kernel_size, 1, 1, 1)

    def forward(self, x, grad):
        s = self.sub_mean(x)
        h = self.head(s)
        mid = torch.cat((h, grad), dim=1) # concat gradient
        mid = self.conv(mid)
        # b1 = self.b1(h)
        b1 = self.b1(mid)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b_out = self.b4(b3)

        res = self.tail(b_out)
        out = self.add_mean(res)
        f_out = out + x

        return f_out
