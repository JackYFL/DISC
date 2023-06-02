# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : food101N.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from .randaugment import TransformFixMatchLarge, TransformFixMatch
import os


class food101N_dataset(Dataset):

    def __init__(self, root_dir, mode):

        self.root_dir = root_dir
        self.mode = mode

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_fixmatch = TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))

        self.imgs_root = root_dir + '/images/'
        train_file = root_dir + '/train.txt'
        test_file = root_dir + '/test.txt'
        imgs_dirs = os.listdir(self.imgs_root)
        imgs_dirs.sort()
        class2idx = {}
        self.train_imgs = []
        self.test_imgs = []

        for idx, food in enumerate(imgs_dirs):
            class2idx[food] = idx

        with open(train_file) as f:
            lines = f.readlines()
            for line in lines:
                train_img = line.strip()
                target = class2idx[train_img.split('/')[0]]
                self.train_imgs.append([train_img, target])
        with open(test_file) as f:
            lines = f.readlines()
            for line in lines:
                test_img = line.strip()
                target = class2idx[test_img.split('/')[0]]
                self.test_imgs.append([test_img, target])

    def __getitem__(self, index):
        if self.mode == 'train':
            img_id, target = self.train_imgs[index]
            img_path = self.imgs_root + img_id + '.jpg'
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target

        elif self.mode == 'train_single':
            img_id, target = self.train_imgs[index]
            img_path = self.imgs_root + img_id + '.jpg'
            image = Image.open(img_path).convert('RGB')
            img = self.transform_train(image)
            return img, target

        elif self.mode == 'train_index':
            ind = index
            img_id, target = self.train_imgs[index]
            img_path = self.imgs_root + img_id + '.jpg'
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, ind

        elif self.mode == 'test':
            ind = index
            img_id, target = self.test_imgs[index]
            img_path = self.imgs_root + img_id + '.jpg'
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'train' or self.mode == 'train_index' or self.mode == 'train_single':
            return len(self.train_imgs)


class food101N_dataloader():

    def __init__(self, root_dir, batch_size, num_workers, num_batches=1000):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.num_batches = num_batches

    def run(self, mode='train'):
        train_dataset = food101N_dataset(root_dir=self.root_dir, mode=mode)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = food101N_dataset(root_dir=self.root_dir, mode='test')
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, test_loader


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    fooddata = food101N_dataset(root_dir='/data/yfli/Food101N/data/food-101/',
                                mode='train')
