import torch.nn as nn
import torch.nn.functional as F
import torch


##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.LeakyReLU(0.2, True),
                  ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # print(x.size())
        # print(skip_input.size())
        x = torch.cat((x, skip_input), 1)
        return x


class AutoEncoder_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AutoEncoder_UNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)

        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True))

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        # print(x.size())
        d1 = self.down1(x)
        # print(d1.size())
        d2 = self.down2(d1)
        # print(d2.size())
        d3 = self.down3(d2)
        # print(d3.size())
        d4 = self.down4(d3)
        # print(d4.size())
        u5 = self.up1(d4, d3)
        # print(u5.size())
        u6 = self.up2(u5, d2)
        # print(u6.size())
        u7 = self.up3(u6)
        # print(u7.size())
        return self.final(u7)


class DecoderNet(nn.Module):
    def __init__(self, in_dim=512, out_channels=3):
        super(DecoderNet, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim, out_channels=256, kernel_size=4, padding=0, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True), )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True), )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, True), )
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def cat(self, x1, x2):
        x1 = self.up1(x1)

        # print(x.size())
        # print(skip_input.size())
        x = torch.cat((x1, x2), 1)
        return x

    def forward(self, features):
        # U-Net generator with skip connections from encoder to decoder
        # print("ouput7",output7.size())
        # u1 = torch.cat((output7, output6), 1) #[none,522,1,1]
        # print("u1",u1.size())
        # print("u2",u2.size())
        # features = features.view(features.size(0), features.size(1), 1, 1)
        u2 = self.up1(features)  # [none,512,4,4]
        u3 = self.up2(u2)  # [none,512,8,8]
        # print("u3",u3.size())
        u4 = self.up3(u3)  # [none,256,16,16]
        # print("u4",u4.size())
        u5 = self.up4(u4)  # [none,64,32,32]
        # print("u5",u5.size())
        out = self.final(u5)
        # print("out",out.size())
        return out


class UUp(nn.Module):
    def __init__(self, in_size, out_size, kernel=4, padding=1, stride=2):
        super(UUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print(x.size())
        x = self.model(x)
        # print(x.size())
        # print(skip_input.size())
        x = torch.cat((x, skip_input), 1)

        return x
