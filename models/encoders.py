# Zhaolun Zou 08/14/2019
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        return

class Resnet18_Encoder(Encoder):
    def __init__(self, pretrained=True):
        super(Resnet18_Encoder, self).__init__()

        pretrained_model = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        del pretrained_model

        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        return

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x', x.shape)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # x = self.conv2(x5)
        # x = self.bn2(x)
        return x1, x2, x3, x4, x5

class Resnet50_Encoder(Encoder):
    def __init__(self, pretrained=True):
        super(Resnet50_Encoder, self).__init__()

        pretrained_model = torchvision.models.resnet50(pretrained=pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        del pretrained_model

        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        return

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('x', x.shape)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # x = self.conv2(x5)
        # x = self.bn2(x)
        return x1, x2, x3, x4, x5
