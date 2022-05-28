import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#########################################
#        SNGAN Generator
#########################################

class ConvTranspose_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=False):
        super(ConvTranspose_block, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.model(input)

class SNGAN_Generator(nn.Module):
    def __init__(self, nz=128, output_nc=3, ngf=64, img_size=128):
        super(SNGAN_Generator, self).__init__()

        self.init_ngf = ngf * 8
        self.input = nn.Linear(nz, 4 * 4 * self.init_ngf)

        model = []

        self.model = nn.Sequential(
            ConvTranspose_block(in_ch=ngf*8, out_ch=ngf*8, kernel_size=4, stride=2, padding=1),
            ConvTranspose_block(in_ch=ngf*8, out_ch=ngf*4, kernel_size=4, stride=2, padding=1),
            ConvTranspose_block(in_ch=ngf*4, out_ch=ngf*2, kernel_size=4, stride=2, padding=1),
            ConvTranspose_block(in_ch=ngf*2, out_ch=ngf, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=ngf, out_channels=output_nc, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.input(x)
        out = out.view(-1, self.init_ngf, 4, 4)
        out = self.model(out)
        return out

# x = torch.randn((5, 128))
# model = SNGAN_Generator(img_size=128)
# preds = model(x)
# print(preds.shape)

#########################################
#        SNGAN Discriminator
#########################################

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=False, norm=True):
        super(Conv_block, self).__init__()

        model = [nn.utils.spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                                  kernel_size=kernel_size, stride=stride,
                                                  padding=padding, bias=bias))]
        if norm:
            model += [nn.BatchNorm2d(out_ch)]
        model += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class SNGAN_Discriminator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, ndf=64, img_size=128):
        super(SNGAN_Discriminator, self).__init__()

        self.ndf = ndf

        assert img_size == 128

        self.model = nn.Sequential(
            Conv_block(in_ch=in_ch, out_ch=ndf, kernel_size=4, stride=2, padding=1, norm=False),
            Conv_block(in_ch=ndf, out_ch=ndf * 2, kernel_size=4, stride=2, padding=1, norm=True),
            Conv_block(in_ch=ndf * 2, out_ch=ndf * 4, kernel_size=4, stride=2, padding=1, norm=True),
            Conv_block(in_ch=ndf * 4, out_ch=ndf * 8, kernel_size=4, stride=2, padding=1, norm=True),
            Conv_block(in_ch=ndf * 8, out_ch=ndf * 8, kernel_size=4, stride=2, padding=1, norm=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=ndf * 8, out_channels=ndf,
                                             kernel_size=4, stride=1, padding=0))
        )

        self.linear = nn.utils.spectral_norm(nn.Linear(ndf, out_ch))

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, self.ndf)
        out = self.linear(out)
        return out

# x = torch.randn((5, 3, 128, 128))
# model = SNGAN_Discriminator()
# preds = model(x)
# print(preds.shape)