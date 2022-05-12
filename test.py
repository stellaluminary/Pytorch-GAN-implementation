import argparse
from configs import options
from data import create_dataset
from models import create_model
from utils import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', '-o',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/cyclegan.yaml')
    args = parser.parse_args()
    opt = options.parser(args)

    opt['Setting']['phase'] = 'test'
    opt['Transforms']['flip'] = False
    opt['Transforms']['rot'] = False
    opt['Transforms']['shuffle'] = False
    opt['Transforms']['resize'] = False

    test_dataset = create_dataset(opt)
    dataset_size = len(test_dataset)
    print('The number of testing images = %d' % dataset_size)

    model = create_model(opt)
    model.print_networks()
    model.load_pretrained_nets()
    model.eval()

    for idx, data in enumerate(test_dataset):

        model.feed_data(data)
        model.forward()

        images = model.get_current_visuals()
        vis_path = model.make_visual_dir(opt['Path']['pretrain_res'])
        util.save_current_imgs(images=images, save_dirs=vis_path, phase=opt['Setting']['phase'],
                                id=idx, min_max=(-1, 1))

if __name__ == '__main__':
    main()