import torch.nn.functional as F
from .unet_parts import *           #.当前文件夹中导入 ..上一次文件夹中导入  啥也没有是从项目的一级目录中导入
class UNet_Origin(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):  #实例化时，如果不对bilinear，新传入值，那么默认是True。即，可选参数
        super(UNet_Origin, self).__init__()         #初始化父类的固定写法
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)#做转置卷积，见李沐讲的
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':        #本项目中主文件不是他，是0_origin_unet_train.py，不是这个Python
    net = UNet_Origin(n_channels=3, n_classes=1)
    print(net)
