import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN, MNIST
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'CIFAR10':
        # if use the orignal full dataset, please uncomment the following
        #dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
        #                  transform=tran_transform)
        # test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
        #                        transform=test_transform)

        # if use the demo images, please uncomment the following
        test_dataset = dset.ImageFolder(os.path.join(args.exp, 'datasets', 'cifar10_demo'), transform=test_transform)
        dataset = -1


    elif config.data.dataset == 'MNIST':
        # if use the orignal full dataset, please uncomment the following
        #dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist'), train=True, download=True,
        #                  transform=tran_transform)
        #test_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist_test'), train=False, download=True,
        #                       transform=test_transform)

        # if use the demo images, please uncomment the following
        test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
           transforms.Resize(config.data.image_size),
           transforms.ToTensor()
        ])
        test_dataset = dset.ImageFolder(os.path.join(args.exp, 'datasets', 'mnist_demo'), transform=test_transform)
        dataset = -1

    elif config.data.dataset == 'CELEBA':
        # if use the orignal full dataset, please uncomment the following
        #if config.data.random_flip:
        #    dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celebA'), split='train',
        #                     transform=transforms.Compose([
        #                         transforms.CenterCrop(140),
        #                         transforms.Resize(config.data.image_size),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),
        #                     ]), download=False)
        #else:
        #    dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celebA'), split='train',
        #                     transform=transforms.Compose([
        #                         transforms.CenterCrop(140),
        #                         transforms.Resize(config.data.image_size),
        #                         transforms.ToTensor(),
        #                     ]), download=False)

        #test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celebA'), split='test',
        #                      transform=transforms.Compose([
        #                          transforms.CenterCrop(140),
        #                          transforms.Resize(config.data.image_size),
        #                          transforms.ToTensor(),
        #                      ]), download=False)

        # if use the demo images, please uncomment the following
        test_augs = transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ])
        test_dataset = dset.ImageFolder(os.path.join(args.exp, 'datasets', 'celeba_demo'), transform=test_augs)
        # test_dataset = dset.ImageFolder(os.path.join(args.exp, 'datasets', 'celeba_ood_demo'), transform=test_augs) # for ood test
        dataset = -1
 
    elif config.data.dataset == "FFHQ":
        # if use the orignal full dataset, please uncomment the following
        # if config.data.random_flip:
        #     dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.ToTensor()
        #     ]), resolution=config.data.image_size)
        # else:
        #     dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
        #                    resolution=config.data.image_size)
        #
        # num_items = len(dataset)
        # indices = list(range(num_items))
        # random_state = np.random.get_state()
        # np.random.seed(2019)
        # np.random.shuffle(indices)
        # np.random.set_state(random_state)
        # train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        # test_dataset = Subset(dataset, test_indices)
        # dataset = Subset(dataset, train_indices)

        # if use the demo images, please uncomment the following
        test_augs = transforms.Compose([
                                  transforms.CenterCrop(256),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ])
        test_dataset = dset.ImageFolder(os.path.join(args.exp, 'datasets', 'ffhq_demo'), transform=test_augs)
        dataset = -1

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.
  
    X = torch.clamp(X, 0.0, 1.0)
    # shape = X.size()
    # X1 = X.squeeze(-1).cpu().numpy()
    # X1 = X1 - np.min(X1)
    # X1 = X1/np.max(X1)
    # X = torch.from_numpy(X1).reshape(shape)
    return X
