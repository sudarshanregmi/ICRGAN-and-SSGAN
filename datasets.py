import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, SubsetRandomSampler
from utils.utils import *


class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_rot1 = transforms.Compose([
                transforms.Resize(args.img_size),
                MyRotationTransform(90),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_rot2 = transforms.Compose([
                transforms.Resize(args.img_size),
                MyRotationTransform(180),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_rot3 = transforms.Compose([
                transforms.Resize(args.img_size),
                MyRotationTransform(270),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            aug_transform = transforms.Compose([
                transforms.Resize(args.img_size+ 8),
                transforms.RandomCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        else:

            dataset_idx = list(range(50000))
            idx = random.sample(dataset_idx, args.data_size)
            sampler = SubsetRandomSampler(idx)

            # shuffled_dataset = torch.utils.data.Subset(Dt(root=args.data_path, train=True, transform=transform, download=True),
            #         idx)
            # self.train = torch.utils.data.DataLoader(shuffled_dataset, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=False,
            #         drop_last=True, pin_memory=True)
            #
            # shuffled_dataset_rot1 = torch.utils.data.Subset(Dt(root=args.data_path, train=True, transform=transform_rot1, download=True),
            #         idx)
            # self.train_rot1 = torch.utils.data.DataLoader(shuffled_dataset_rot1, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=False,
            #         drop_last=True, pin_memory=True)
            #
            # shuffled_dataset_rot2 = torch.utils.data.Subset(Dt(root=args.data_path, train=True, transform=transform_rot2, download=True),
            #         idx)
            # self.train_rot2 = torch.utils.data.DataLoader(shuffled_dataset_rot2, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=False,
            #         drop_last=True, pin_memory=True)
            #
            # shuffled_dataset_rot3 = torch.utils.data.Subset(Dt(root=args.data_path, train=True, transform=transform_rot3, download=True),
            #         idx)
            # self.train_rot3 = torch.utils.data.DataLoader(shuffled_dataset_rot3, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=False,
            #         drop_last=True, pin_memory=True)

            # aug_shuffled_dataset = torch.utils.data.Subset(Dt(root=args.data_path, train=True, transform=aug_transform, download=True),
            #         idx)
            # self.aug_train = torch.utils.data.DataLoader(aug_shuffled_dataset, batch_size=args.dis_batch_size, num_workers=args.num_workers, shuffle=False,
            #         drop_last=True, pin_memory=True)

            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=False, drop_last=True, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True)

            # self.aug_train = torch.utils.data.DataLoader(
            #     Dt(root=args.data_path, train=True, transform=aug_transform, download=True),
            #     batch_size=args.dis_batch_size, shuffle=False, drop_last=True, sampler=sampler,
            #     num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
