import math
import torch
import torch.nn as nn

#########################################
#        WGAN Generator
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

class WGAN_Generator(nn.Module):
    def __init__(self, nz=100, output_nc=3, ngf=64, img_size=128):
        super(WGAN_Generator, self).__init__()

        self.nz = nz

        ch = ngf * (img_size // 8)
        model = [ConvTranspose_block(in_ch=nz, out_ch=ch, kernel_size=4,
                                     stride=1, padding=0, bias=False)]

        n_upsampling = int(math.log2(img_size // 8))
        for i in range(n_upsampling):
            model += [ConvTranspose_block(in_ch=ch, out_ch=ch//2, kernel_size=4,
                                          stride=2, padding=1, bias=False)]
            ch = ch // 2

        model += [nn.ConvTranspose2d(in_channels=ngf, out_channels=output_nc, kernel_size=4,
                                     stride=2, padding=1, bias=False),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        return self.model(x)

# x = torch.randn((5, 100))
# model = WGAN_Generator(img_size=128)
# preds = model(x)
# print(preds.shape)

#########################################
#        WGAN Discriminator
#########################################

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=False, norm=True):
        super(Conv_block, self).__init__()

        model = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                           kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=bias)]
        if norm:
            model += [nn.BatchNorm2d(out_ch)]
        model += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class WGAN_Discriminator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, ndf=64, img_size=128):
        super(WGAN_Discriminator, self).__init__()

        ch = ndf
        model = [Conv_block(in_ch=in_ch, out_ch=ndf, kernel_size=4,
                            stride=2, padding=1, bias=False, norm=False)]

        n_dowmsampling = int(math.log2(img_size // 8))
        for i in range(n_dowmsampling):
            model += [Conv_block(in_ch=ch, out_ch=ch * 2, kernel_size=4,
                                stride=2, padding=1, bias=False, norm=True)]
            ch *= 2

        model += [nn.Conv2d(in_channels=ch, out_channels=out_ch, kernel_size=4,
                            stride=1, padding=0, bias=False)
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        return out


# x = torch.randn((5, 3, 128, 128))
# model = WGAN_Discriminator()
# preds = model(x)
# print(preds.shape)