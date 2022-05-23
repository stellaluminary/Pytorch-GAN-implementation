import torch
import torch.nn as nn
from torch.nn import functional as F

#########################################
#        SNGAN Generator
#########################################

class ResBlock_UP(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu, upsample=False):
        super(ResBlock_UP, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        x = input
        return self.upsample(x) + self.model(x)

class SNGAN_Generator(nn.Module):
    def __init__(self, nz=128, output_nc=3, ngf=64, img_size=128):
        super(SNGAN_Generator, self).__init__()

        # assume nz=128, ngf=64, ngf*8=512, img_size=128
        self.nz = nz
        self.init_ngf = ngf * 8

        self.input = nn.Linear(nz, 4*4*self.init_ngf)
        self.model = nn.Sequential(
            ResBlock_UP(in_ch=ngf*8, out_ch=ngf*8),
            ResBlock_UP(in_ch=ngf*8, out_ch=ngf*8),
            ResBlock_UP(in_ch=ngf*8, out_ch=ngf*4),
            ResBlock_UP(in_ch=ngf*4, out_ch=ngf*2),
            ResBlock_UP(in_ch=ngf*2, out_ch=ngf),
        )

        last_block = [nn.BatchNorm2d(ngf),
                      nn.ReLU(),
                      nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1),
                      nn.Tanh()]

        self.last_block = nn.Sequential(*last_block)

    def forward(self, x):
        x = self.input(x)
        x = x.view(-1, self.init_ngf, 4, 4)
        # [batch, 64, 128, 128]
        x = self.model(x)
        x = self.last_block(x)
        return x

#x = torch.randn((5, 128))
#model = SNGAN_Generator(img_size=128)
#preds = model(x)
#print(model)
#print(preds.shape)

#########################################
#        SNGAN Discriminator
#########################################

class ResBlock_Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, downsample=True):
        super(ResBlock_Down, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                             kernel_size=kernel_size, padding=padding))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                                             kernel_size=kernel_size, padding=padding))

        self.conv_shortcut = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))

    def forward(self, input):
        out = input

        out = F.relu(out)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)

        skip_out = self.conv_shortcut(input)
        if self.downsample:
            skip_out = F.avg_pool2d(skip_out, 2)

        return out + skip_out

class SNGAN_PorjectionDiscriminator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, ndf=64, img_size=128):
        super(SNGAN_PorjectionDiscriminator, self).__init__()

        # assume ndf=64, img_size=128
        self.ndf = ndf

        self.pre_model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=ndf,
                                             kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=ndf, out_channels=ndf,
                                             kernel_size=3, padding=1)),
            nn.AvgPool2d(2)
        )

        self.pre_skip = nn.utils.spectral_norm(nn.Conv2d(in_ch, ndf, kernel_size=1, padding=0))


        model = [
            ResBlock_Down(in_ch=ndf, out_ch=ndf * 2),
            ResBlock_Down(in_ch=ndf * 2, out_ch=ndf * 4),
            ResBlock_Down(in_ch=ndf * 4, out_ch=ndf * 8),
            ResBlock_Down(in_ch=ndf * 8, out_ch=ndf * 8),
            ResBlock_Down(in_ch=ndf * 8, out_ch=ndf * 8, downsample=False)
        ]

        self.model = nn.Sequential(*model)

        self.linear = nn.Linear(ndf * 8, out_ch)

    def forward(self, x):
        # out = [batch, 3, 128, 128] -> [batch, 64, 64, 64]
        out = self.pre_model(x)
        out = out + self.pre_skip(F.avg_pool2d(x, 2))
        # out = [batch, 3, 128, 128] -> [batch, 512, 4, 4]
        out = self.model(out)
        out = F.relu(out)
        # out = [batch, 512, 4, 4] -> [batch, 512]
        out = torch.sum(out, dim=(2,3))
        # out = [batch, 512] -> [batch, 1]
        out = self.linear(out)
        return out

#x = torch.randn((5, 3, 128, 128))
#model = SNGAN_PorjectionDiscriminator()
#preds = model(x)
#print(preds.shape)