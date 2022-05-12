import torch
import torch.nn as nn
from torch.nn import init

#########################################
#        CycleGAN Generator
#########################################

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type=nn.ReflectionPad2d, norm_layer=nn.InstanceNorm2d):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type=padding_type, norm_layer=norm_layer)

    def build_conv_block(self, dim, padding_type, norm_layer):
        p = 0

        conv_block = [
            padding_type(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        conv_block += [
            padding_type(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)  # skip connections

class Resnet_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9, padding_mode='reflect'):
        super(Resnet_Generator, self).__init__()

        if padding_mode == 'reflect':
            padding_type = nn.ReflectionPad2d

        # 256 -> (256 + 3*2(reflectionpad) - 7(kernel_size)) / 1(stride) + 1 = 256
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(inplace=True)
                 ]

        n_downsampling = 2
        # 256 -> 128 -> 64
        for i in range(n_downsampling):
            mult = 2 ** i # 1, 2
            model += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(ngf*mult*2),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** n_downsampling # 4
        # Cx64x64 -> Cx64x64
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]

        #64 -> 128 -> 256
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling-i)
            # ConvTranspose res = (h-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult//2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf*mult//2)),
                      nn.ReLU(inplace=True)]

        # 256 -> (256 + 3*2(reflectionpad) - 7(kernel_size)) / 1(stride) + 1 = 256
        model += [padding_type(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# x = torch.randn((5, 3, 256, 256))
# model = Resnet_Generator(3,3)
# preds = model(x)
# print(preds.shape)

#########################################
#        CycleGAN Discriminator
#########################################

class Discriminator_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, normalize=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=True
                      ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PatchGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], norm_layer=nn.InstanceNorm2d):
        super(PatchGAN_Discriminator, self).__init__()

        layers = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=True,
                      ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Discriminator_Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # results = [256x256] -> [30x30]
        return self.model(x)


# x = torch.randn((5, 3, 256, 256))
# model = PatchGAN_Discriminator(in_channels=3)
# preds = model(x)
# print(preds.shape)