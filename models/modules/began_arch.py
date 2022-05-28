import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
#        BEGAN Decoder (Generator)
#########################################

class Deconv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deconv_block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, x):
        return self.model(x)

class BEGAN_Decoder(nn.Module):
    def __init__(self, hidden=64, output_nc=3, ngf=64, img_size=128):
        super(BEGAN_Decoder, self).__init__()

        self.h = hidden
        self.init_ngf = ngf

        self.first_fc = nn.Linear(hidden, 8*8*ngf)

        self.model = nn.Sequential(
            Deconv_block(ngf, ngf),
            Deconv_block(ngf, ngf),
            Deconv_block(ngf, ngf),
            Deconv_block(ngf, ngf),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.first_fc(x)
        x = x.view(-1, self.init_ngf, 8, 8)
        x = self.model(x)
        x = self.last_block(x)
        return x

# hidden = 64
# x = torch.randn((5, hidden))
# model = BEGAN_Decoder(hidden=hidden, img_size=128)
# preds = model(x)
# print(preds.shape)

#########################################
#        BEGAN Encoder
#########################################

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            # In BEGAN paper, there is no other explanation about sub-sampling except stride=2
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

class BEGAN_Encoder(nn.Module):
    def __init__(self, input_nc=3, hidden=128, ndf=128, img_size=128):
        super(BEGAN_Encoder, self).__init__()

        self.init_ndf = ndf

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            Conv_block(ndf, ndf),
            Conv_block(ndf, ndf * 2),
            Conv_block(ndf * 2, ndf * 3),
            Conv_block(ndf * 3, ndf * 4),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True)
        )

        self.last_fc = nn.Linear(8 * 8 * 4 * ndf, hidden)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_block(x)
        x = self.last_block(x)
        x = x.view(-1, self.init_ndf*4*8*8)
        x = self.last_fc(x)
        return x

# x = torch.randn((5, 3, 128, 128))
# model = BEGAN_Encoder()
# preds = model(x)
# print(preds.shape)

#########################################
#        BEGAN Discriminator
#########################################

class BEGAN_Discriminator(nn.Module):
    def __init__(self, in_ch=3, hidden=64, out_ch=3, ndf=128, ngf=128, img_size=128):
        super(BEGAN_Discriminator, self).__init__()

        self.encoder = BEGAN_Encoder(input_nc=in_ch, hidden=hidden, ndf=ndf)
        self.decoder = BEGAN_Decoder(hidden=hidden, output_nc=out_ch, ngf=ngf)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

# x = torch.randn((5, 3, 128, 128))
# model = BEGAN_Discriminator()
# preds = model(x)
# print(preds.shape)