import logging

logger = logging.getLogger('base')

def create_model(opt):
    model = opt['Model_Param']['model_name']

    if model == 'cyclegan':
        from .cyclegan_model import CycleGANModel as M
    elif model == 'dcgan':
        from .dcgan_model import DCGANModel as M
    elif model == 'wgan':
        from .wgan_model import WGANModel as M
    elif model == 'wgan-gp':
        from .wgan_gp_model import WGANGPModel as M
    elif model == 'sngan':
        from .sngan_model import SNGANModel as M
    elif model == 'sagan':
        from .sagan_model import SAGANModel as M

    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    instance = M(opt)
    logger.info('Model [%s] is created.' % type(instance).__name__)
    return instance

