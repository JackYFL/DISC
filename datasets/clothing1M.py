# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : Clothing1M.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
import torch
from .randaugment import TransformFixMatchLarge
import os


class clothing_dataset(Dataset):

    def __init__(self, root_dir, mode, num_samples=0):

        self.root_dir = root_dir
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.num_samples = num_samples
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        # import ipdb; ipdb.set_trace()
        self.transform_fixmatch = TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
        self.index_dic = {}
        with open('%s/label/noisy_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0]
                if os.path.exists(self.root_dir + '/' + img_path):
                    self.train_labels[img_path] = int(entry[1])

        with open('%s/label/clean_label_kv.txt' % self.root_dir, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0]
                if os.path.exists(self.root_dir + '/' + img_path):
                    self.test_labels[img_path] = int(entry[1])

        if mode == 'train' or mode == 'all':
            self.train_imgs = []
            self.labels = torch.zeros(len(self.train_labels)).long()
            with open('%s/label/noisy_train_key_list.txt' % self.root_dir,
                      'r') as f:
                lines = f.read().splitlines()
                for i, l in enumerate(lines):
                    img_path = l.strip()
                    self.index_dic[img_path] = i
                    self.labels[i] = torch.tensor(
                        self.train_labels[img_path]).long()
                    self.train_imgs.append(img_path)

        elif mode == 'test' or mode == 'test_ind':
            self.test_imgs = []
            with open('%s/label/clean_test_key_list.txt' % self.root_dir,
                      'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = l.strip()
                    self.test_imgs.append(img_path)

        elif mode == 'val':
            self.val_imgs = []
            with open('%s/label/clean_val_key_list.txt' % self.root_dir,
                      'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = l.strip()
                    self.val_imgs.append(img_path)

    def sample_subset(self, num_class=14):  #sample a class-balanced subset
        random.shuffle(self.train_imgs)
        class_num = torch.zeros(num_class)
        self.train_imgs_subset = []
        for impath in self.train_imgs:
            label = self.train_labels[impath]
            if class_num[label] < (self.num_samples / 14) and len(
                    self.train_imgs_subset) < self.num_samples:
                self.train_imgs_subset.append(impath)
                class_num[label] += 1
        random.shuffle(self.train_imgs_subset)
        return

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs_subset[index]
            ind = self.index_dic[img_path]
            target = self.train_labels[img_path]
            img_path = self.root_dir + '/' + img_path
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, ind
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            ind = self.index_dic[img_path]
            # ind = index
            target = self.train_labels[img_path]
            img_path = self.root_dir + '/' + img_path
            image = Image.open(img_path).convert('RGB')
            img = self.transform_fixmatch(image)
            return img, target, ind
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            img_path = self.root_dir + '/' + self.test_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target
        elif self.mode == 'test_ind':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            img_path = self.root_dir + '/' + self.test_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target, img_path
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            img_path = self.root_dir + '/' + self.val_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform_test(image)
            return img, target

    def __len__(self):
        if self.mode == 'test' or self.mode == 'test_ind':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)
        elif self.mode == 'train':
            return len(self.train_imgs_subset)
        elif self.mode == 'all':
            return len(self.train_imgs)


class clothing_dataloader():
    def __init__(self, root_dir, batch_size, num_workers, num_batches=1000):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.num_batches = num_batches
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])

    def run_all(self):
        train_dataset = clothing_dataset(root_dir=self.root_dir,
                                         mode='all',
                                         num_samples=self.num_batches *
                                         self.batch_size)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = clothing_dataset(root_dir=self.root_dir,
                                        mode='test',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, test_loader

    def run(self):
        train_dataset = clothing_dataset(root_dir=self.root_dir,
                                         mode='train',
                                         num_samples=self.num_batches *
                                         self.batch_size)
        train_dataset.sample_subset()
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = clothing_dataset(root_dir=self.root_dir,
                                        mode='test',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        eval_dataset = clothing_dataset(root_dir=self.root_dir,
                                        mode='val',
                                        num_samples=self.num_batches *
                                        self.batch_size)
        eval_loader = DataLoader(dataset=eval_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        return train_loader, eval_loader, test_loader


if __name__ == '__main__':
    clothing = clothing_dataloader(root_dir='/data/yfli/Clothing1M',
                                   batch_size=128,
                                   num_workers=4)
    train_loader, eval_loader, test_loader = clothing.run()
    for img, targets, index in train_loader:
        pass