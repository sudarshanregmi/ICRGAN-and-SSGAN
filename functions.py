# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.autograd import Variable
from imageio import imsave
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import logging
import ramps

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

import wandb

logger = logging.getLogger(__name__)
mse_loss = nn.MSELoss()


def get_current_consistency_weight(args, epoch, step_in_epoch, total_steps_in_epoch):
    epoch = epoch - args.consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return args.lmbda * ramps.sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts)


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])

    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / ( global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1-alpha, param.data)

def augment(img, type='default'):
    if type == 'default':
        if torch.randn(1).item() > 0:
            # horizontal flip
            img = img.flip(-1)
        # shifting pixels
        i = torch.randint(-4, 5, (1,)).item()
        j = torch.randint(-4, 5, (1,)).item()
        img = torch.roll(img, (i, j), (-1, -2))
        return img
    elif type == 'resize_crop':
        # resize to 40x40
        img.resize_(img.shape[0], img.shape[1], img.shape[2] + 8, img.shape[3] + 8)
        # crop to 32x32
        x = torch.randint(0, 9, (1,)).item()
        y = torch.randint(0, 9, (1,)).item()
        return img[:, :, x:x + 32, y:y + 32]
    elif type == 'horizontal_flip':
        # horizontal flip
        img = img.flip(-1)
        return img
    elif type == 'pixel_shift':
        # shifting pixels
        i = torch.randint(-4, 5, (1,)).item()
        j = torch.randint(-4, 5, (1,)).item()
        img = torch.roll(img, (i, j), (-1, -2))
        return img


def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        dis_optimizer.zero_grad()

        d_real_class_loss = 0
        g_fake_class_loss = 0

        real_imgs = imgs.type(torch.cuda.FloatTensor)
        fake_imgs = gen_net(z).detach()

        if args.ssup:

            x = real_imgs 
            x_90 = x.transpose(2,3)
            x_180 = x.flip(2,3)
            x_270 = x.transpose(2,3).flip(2,3)
            real_imgs = torch.cat((x,x_90,x_180,x_270),0)

            x = fake_imgs
            x_90 = x.transpose(2,3)
            x_180 = x.flip(2,3)
            x_270 = x.transpose(2,3).flip(2,3)
            fake_imgs = torch.cat((x, x_90, x_180, x_270),0)

        assert fake_imgs.size() == real_imgs.size()
        real_validity, d_real_rot_logits, d_real_rot_prob = dis_net(real_imgs)
        fake_validity, g_fake_rot_logits, g_fake_rot_prob = dis_net(fake_imgs)

        D_x = real_validity.mean().item()
        D_z_1 = fake_validity.mean().item()

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))

        if args.ssup:
            # Add auxiiary rotation loss
            rot_labels = torch.zeros(4*args.dis_batch_size).cuda()
            for i in range(4*args.dis_batch_size):
                if i < args.dis_batch_size:
                    rot_labels[i] = 0
                elif i < 2*args.dis_batch_size:
                    rot_labels[i] = 1
                elif i < 3*args.dis_batch_size:
                    rot_labels[i] = 2
                else:
                    rot_labels[i] = 3
            
            # rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
            rot_labels = rot_labels.to(torch.int64)
            d_real_class_loss = torch.mean(F.cross_entropy(
                                        input = d_real_rot_logits,
                                        target = rot_labels))

            d_loss += args.rot_weight_dis * d_real_class_loss

        if args.bcr:
            # augment real images
            aug_real_imgs = augment(real_imgs, args.augment_type)
            aug_real_validity, d_aug_real_rot_logits, d_aug_real_rot_prob = dis_net(aug_real_imgs)
            # augment fake images
            aug_fake_imgs = augment(fake_imgs, args.augment_type)
            aug_fake_validity, d_aug_fake_rot_logits, d_aug_fake_rot_prob = dis_net(aug_fake_imgs)
            L_real = args.lmbda_real * mse_loss(real_validity, aug_real_validity)
            L_fake = args.lmbda_fake * mse_loss(fake_validity, aug_fake_validity)
            d_loss += L_real + L_fake

        if args.zcr:
            noise = torch.cuda.FloatTensor(np.random.normal(0, args.lmbda_noise, (imgs.shape[0], args.latent_dim)))
            #augment latent space
            aug_z = z + noise
            fake_aug_imgs = gen_net(aug_z).detach()

            if args.ssup:
                x = fake_aug_imgs
                x_90 = x.transpose(2,3)
                x_180 = x.flip(2,3)
                x_270 = x.transpose(2,3).flip(2,3)
                fake_aug_imgs = torch.cat((x, x_90, x_180, x_270),0)

            fake_aug_validity, g_aug_fake_rot_logits, g_aug_fake_rot_prob = dis_net(fake_aug_imgs)
            L_dis = args.lmbda_dis * mse_loss(fake_validity, fake_aug_validity)
            d_loss += L_dis

        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)

            if args.ssup:
                x = gen_imgs
                x_90 = x.transpose(2,3)
                x_180 = x.flip(2,3)
                x_270 = x.transpose(2,3).flip(2,3)
                gen_imgs = torch.cat((x, x_90, x_180, x_270),0)

            fake_validity, g_fake_rot_logits, g_fake_rot_prob = dis_net(gen_imgs)
            D_z_2 = fake_validity.mean().item()
            g_loss = -torch.mean(fake_validity)

            if args.zcr:

                noise = torch.cuda.FloatTensor(np.random.normal(0, args.lmbda_noise, (args.gen_batch_size, args.latent_dim)))
                aug_z = gen_z + noise
                fake_aug_imgs = gen_net(aug_z)

                if args.ssup:
                    x = fake_aug_imgs
                    x_90 = x.transpose(2,3)
                    x_180 = x.flip(2,3)
                    x_270 = x.transpose(2,3).flip(2,3)
                    fake_aug_imgs = torch.cat((x, x_90, x_180, x_270),0)

                L_gen = - args.lmbda_gen * mse_loss(gen_imgs, fake_aug_imgs)
                g_loss += L_gen

            if args.ssup:
                rot_labels = torch.zeros(4*args.gen_batch_size,).cuda()
                for i in range(4*args.gen_batch_size):
                    if i < args.gen_batch_size:
                        rot_labels[i] = 0
                    elif i < 2*args.gen_batch_size:
                        rot_labels[i] = 1
                    elif i < 3*args.gen_batch_size:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3
                
                # rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                rot_labels = rot_labels.to(torch.int64)
                g_fake_class_loss = torch.mean(F.cross_entropy(
                    input = g_fake_rot_logits, 
                    target = rot_labels))
        
                g_loss += args.rot_weight_gen * g_fake_class_loss

            g_loss.backward()
            gen_optimizer.step()

            wandb.log({
                'D(x)': D_x,
                'D(G(z1))': D_z_1, 
                'D(G(z2))': D_z_2, 
                'D loss': d_loss.item(),
                'G loss': g_loss.item(),
                'G Fake Class loss': g_fake_class_loss,
                'D Real Class loss': d_real_class_loss
            })

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    # logger.info('=> calculate inception score')
    # mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    # os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    # writer.add_scalar('Inception_score/mean', mean, global_steps)
    # writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
