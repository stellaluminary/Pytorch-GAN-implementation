import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1
                                     or classname.find('ConvTranspose') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

####################
# define network
####################

# Generator
def define_G(opt, name, device=None):
    gpu_ids = opt['Setting']['gpu_ids']
    opt_net = opt['Model_Param']
    model_name = opt_net['model_name']

    if model_name in ['cyclegan']:
        from models.modules.cyclegan_arch import Resnet_Generator
        netG = Resnet_Generator(input_nc=opt_net['input_nc'],
                                              output_nc=opt_net['output_nc'],
                                              ngf=opt_net['ngf'], norm_layer=nn.InstanceNorm2d,
                                              n_blocks=9, padding_mode='reflect')
    elif model_name in ['dcgan']:
        from models.modules.dcgan_arch import DCGAN_Generator
        netG = DCGAN_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                               ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['wgan']:
        from models.modules.wgan_arch import WGAN_Generator
        netG = WGAN_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                              ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['wgan-gp']:
        from models.modules.wgan_gp_arch import WGAN_GP_Generator
        netG = WGAN_GP_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                                 ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['began']:
        from models.modules.began_arch import BEGAN_Decoder
        netG = BEGAN_Decoder(hidden=opt_net['nh'], output_nc=opt_net['output_nc'],
                                 ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['sngan']:
        if opt_net['model_type'] == 'resblock':
            from models.modules.sngan_res_arch import SNGAN_Generator
            netG = SNGAN_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                                   ngf=opt_net['ngf'], img_size=opt_net['img_size'])
        elif opt_net['model_type'] == 'dcgan':
            from models.modules.sngan_arch import SNGAN_Generator
            netG = SNGAN_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                                   ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['sagan']:
        from models.modules.sagan_arch import SAGAN_Generator
        netG = SAGAN_Generator(nz=opt_net['nz'], output_nc=opt_net['output_nc'],
                               ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(model_name))

    if opt['Setting']['phase'] == 'train':
        init_weights(netG, init_type=opt_net['init_type'], init_gain=opt_net['init_gain'])
        print('Initialize %s with %s' % (name, opt_net['init_type']))
    if gpu_ids:
        assert torch.cuda.is_available()
        if device is not None:
            netG = nn.DataParallel(netG.to(device))
        else:
            netG = nn.DataParallel(netG)
    return netG

# Discriminator
def define_D(opt, name):
    #gpu_ids = opt['Setting']['gpu_ids']
    opt_net = opt['Model_Param']
    model_name = opt_net['model_name']

    if model_name in ['cyclegan']:
        from models.modules.cyclegan_arch import PatchGAN_Discriminator
        netD = PatchGAN_Discriminator(in_channels=opt_net['input_nc'],
                                                    features=[64, 128, 256, 512],
                                                    norm_layer=nn.InstanceNorm2d)
    elif model_name in ['dcgan']:
        from models.modules.dcgan_arch import DCGAN_Discriminator
        netD = DCGAN_Discriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                              ndf=opt_net['ndf'], img_size=opt_net['img_size'])
    elif model_name in ['wgan']:
        from models.modules.wgan_arch import WGAN_Discriminator
        netD = WGAN_Discriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                              ndf=opt_net['ndf'], img_size=opt_net['img_size'])
    elif model_name in ['wgan-gp']:
        from models.modules.wgan_gp_arch import WGAN_GP_Discriminator
        netD = WGAN_GP_Discriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                            ndf=opt_net['ndf'], img_size=opt_net['img_size'])
    elif model_name in ['began']:
        from models.modules.began_arch import BEGAN_Discriminator
        netD = BEGAN_Discriminator(in_ch=opt_net['input_nc'], hidden=opt_net['nh'], out_ch=opt_net['output_nc'],
                                   ndf=opt_net['ndf'], ngf=opt_net['ngf'], img_size=opt_net['img_size'])
    elif model_name in ['sngan']:
            if opt_net['model_type'] == 'resblock':
                from models.modules.sngan_res_arch import SNGAN_PorjectionDiscriminator
                netD = SNGAN_PorjectionDiscriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                                     ndf=opt_net['ndf'], img_size=opt_net['img_size'])
            elif opt_net['model_type'] == 'dcgan':
                from models.modules.sngan_arch import SNGAN_Discriminator
                netD = SNGAN_Discriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                           ndf=opt_net['ndf'], img_size=opt_net['img_size'])
    elif model_name in ['sagan']:
        from models.modules.sagan_arch import SAGAN_ProjectionDiscriminator
        netD = SAGAN_ProjectionDiscriminator(in_ch=opt_net['input_nc'], out_ch=1,
                                   ndf=opt_net['ndf'], img_size=opt_net['img_size'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(model_name))

    init_weights(netD, init_type=opt_net['init_type'], init_gain=opt_net['init_gain'])
    print('Initialize %s with %s' % (name, opt_net['init_type']))

    netD = nn.DataParallel(netD)
    return netD