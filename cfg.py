# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    random_seed = random.randint(1, 10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssup', type=str2bool, default=False)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--resnet', type=str2bool, default=True)
    parser.add_argument('--spectral_normed', type=str2bool, default=True)
    parser.add_argument('--rot_weight_gen', type=float, default=0.5)
    parser.add_argument('--rot_weight_dis', type=float, default=1.0)
    parser.add_argument('--num_rotation', type=int, default=4)
    parser.add_argument('--bcr', type=str2bool, default=False)
    parser.add_argument('--zcr', type=str2bool, default=False)
    parser.add_argument('--augment_type', type=str, default='default', 
            choices=['crop_resize', 'horizontal_flip', 'shift_pixel', 'default'])
    parser.add_argument(
        '--lmbda_fake',
        type=float,
        default=10.0,
        help='coefficient lmbda fake value')
    parser.add_argument(
        '--lmbda_real',
        type=float,
        default=10.0,
        help='coefficient lmbda real value')
    parser.add_argument(
        '--lmbda_noise',
        type=float,
        default=0.1,
        help='coefficient lmbda disc value')
    parser.add_argument(
        '--lmbda_gen',
        type=float,
        default=0.5,
        help='coefficient lmbda gen value')
    parser.add_argument(
        '--lmbda_dis',
        type=float,
        default=30.0,    # [20, 30]
        help='coefficient lmbda disc value')
    parser.add_argument(
        '--consistency_rampup_starts',
        type=int,
        default=0,
        help='epoch when consistency rampup starts')
    parser.add_argument(
        '--data_size',
        type=int,
        default=5000,
        help='Size of data to train on')
    parser.add_argument(
        '--consistency_rampup_ends',
        type=int,
        default=1000,
        help='epoch when consistency rampup ends')
    parser.add_argument(
        '--lmbda',
        type=float,
        default=1e-6,
        help='weighing factor of ICT')
    parser.add_argument(
        '--lmbda_type',
        type=str,
        default='fixed',
        choices=['fixed', 'rampup'],
        help='weighing type for IC loss')
    parser.add_argument(
        '--model_type',
        type=str,
        default='single',
        choices=['single', 'mean_teacher'],
        help='type of model for estimated model average')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=50000,
        help='set the max iteration number')
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=32,
        help='size of the batches')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='size of each image dimension')
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    parser.add_argument(
        '--n_critic',
        type=int,
        default=5,
        help='number of training steps for discriminator per iter')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=50,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp_name',
        type=str,
        default='sngan_cifar_{}'.format(random_seed),
        help='The name of exp')
    parser.add_argument(
        '--d_spectral_norm',
        type=str2bool,
        default=True,
        help='add spectral_norm on discriminator?')
    parser.add_argument(
        '--g_spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=256,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=128,
                        help='The base channel num of disc')
    parser.add_argument(
        '--model',
        type=str,
        default='sngan_cifar10',
        help='path of model')
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument('--random_seed', type=int, default=random_seed)

    opt = parser.parse_args()
    return opt
