import logging

logger = logging.getLogger('base')

def create_model(opt):
    model = opt['Model_Param']['model_name']

    if model == 'cyclegan':
        from .cyclegan_model import CycleGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    instance = M(opt)
    logger.info('Model [%s] is created.' % type(instance).__name__)
    return instance

