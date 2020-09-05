# full assembly of the sub-parts to form the complete net
from models.unet_parts import *
class UNet(nn.Module):
    def __init__(self, n_channels=3): #n_classes=3
        super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        # print('n_channels', n_channels)
        # self.down1 = down(n_channels, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = x        #
        # x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) #, up_x1
        x = self.up2(x, x3) #, up_x2
        x = self.up3(x, x2) #, up_x3
        x = self.up4(x, x1) #, up_x4
        # p1 = up_x1 - x4
        # p2 = up_x2 - x3
        # p3 = up_x3 - x2
        # p4 = up_x4 - x1
        # semantic_out = self.outc(x)
        return x #semantic_out
        # return x1

class UNet2(nn.Module):
    def __init__(self, n_channels=3): #n_classes=3
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        # print('n_channels', n_channels)
        # self.down1 = down(n_channels, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)

    def forward(self, x):
        # x1 = x        #
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # semantic_out = self.outc(x)
        return x #semantic_out

class UNet_con(nn.Module):
    def __init__(self, n_channels=3): #n_classes=3
        super(UNet_con, self).__init__()
        # self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        # print('n_channels', n_channels)
        # self.down1 = down(n_channels, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        self.up1 = up(1536, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)

    def forward(self, x, feature):
        x1 = x        #
        # x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = torch.cat((x5, feature), 1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # semantic_out = self.outc(x)
        return x #semantic_out