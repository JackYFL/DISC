# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : Anmimal10N.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from .randaugment import TransformFixMatchMedium
import os


class animal10N_dataset(Dataset):

    def __init__(self, root_dir, mode):

        self.root_dir = root_dir
        self.mode = mode
        self.transform_test = transforms.Compose([
            # transforms.Resize(64),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_fixmatch = TransformFixMatchMedium(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))

        self.train_dir = root_dir + '/training/'
        self.test_dir = root_dir + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)
        self.train_imgs = []
        self.test_imgs = []

        for img in train_imgs:
            self.train_imgs.append([img, int(img[0])])
        for img in test_imgs:
            self.test_imgs.append([img, int(img[0])])

    def __getitem__(self, index):
        if self.mode == 'train':
            img_id, target = self.train_imgs[index]
            img_path = self.train_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target

        elif self.mode == 'train_single':
            img_id, target = self.train_imgs[index]
            img_path = self.train_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_train(image)
            return img, target

        elif self.mode == 'train_index':
            ind = index
            img_id, target = self.train_imgs[index]
            img_path = self.train_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, ind

        elif self.mode == 'test':
            ind = index
            img_id, target = self.test_imgs[index]
            img_path = self.test_dir + img_id
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'train' or self.mode == 'train_index' or self.mode == 'train_single':
            return len(self.train_imgs)


class animal10N_dataloader():

    def __init__(self, root_dir, batch_size, num_workers, num_batches=1000):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.num_batches = num_batches

    def run(self, mode='train'):
        train_dataset = animal10N_dataset(root_dir=self.root_dir, mode=mode)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = animal10N_dataset(root_dir=self.root_dir, mode='test')
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, test_loader


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    fooddata = animal10N_dataset(root_dir='/data/yfli/Animal1N/', mode='train')