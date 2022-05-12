import os
import os.path as osp
import logging
import yaml
import torch

def parser(parse_args, is_train=True):

    with open(parse_args.filename, 'r') as f:
        try:
            opt = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    if opt['Setting']['phase'] == 'train':
        opt['is_train'] = True
    else:
        opt['is_train'] = False

    # choose the device in torch version
    opt['device'] = torch.device('cuda' if opt['Setting']['gpu_ids'] else 'cpu')

    # show the gpu_list based on configuration
    gpu_list = ','.join(str(x) for x in opt['Setting']['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('Export CUDA_VISIBLE_DEVICES = ' + gpu_list)
    print('Running Device :', opt['device'])

    return opt

def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(state_idx))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(state_idx))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])