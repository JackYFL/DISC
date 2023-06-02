# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : webvision.py

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import re
from .randaugment import TransformFixMatchLarge, TransformFixMatchMax


class imagenet_dataset(Dataset):

    def __init__(self, root_dir, num_class):
        self.root = root_dir + '/val/'
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])

        self.val_data = []
        imgnetval_dirs = os.listdir(self.root)
        imgnetval_dirs.sort(key=lambda i: int(re.findall(r'(\d+)', i)[0]))
        val_dirs = imgnetval_dirs[:num_class]
        for c, imgnet_dir in enumerate(val_dirs):
            imgs_dir = self.root + imgnet_dir
            imgs = os.listdir(imgs_dir)
            for img in imgs:
                self.val_data.append(
                    [c, os.path.join(self.root, imgnet_dir, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)


class webvision_dataset(Dataset):

    def __init__(self, root_dir, mode, num_class):
        self.root = root_dir
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_fixmatch = TransformFixMatchLarge(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))

        if self.mode == 'test':
            self.val_imgs = []
            self.val_labels = {}
            with open(self.root + 'info/val_filelist.txt') as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < num_class:
                        self.val_imgs.append(img)
                        self.val_labels[img] = target
        else:
            self.train_imgs = []
            self.train_labels = {}
            with open(self.root + 'info/train_filelist_google.txt') as f:
                lines = f.readlines()
                for line in lines:
                    img, target = line.split()
                    target = int(target)
                    if target < num_class:
                        self.train_imgs.append(img)
                        self.train_labels[img] = target

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(self.root + img_path).convert('RGB')
            img_aug = self.transform_fixmatch(image)
            return img_aug, target, index

        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(self.root + 'val_images_256/' +
                               img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class webvision_dataloader():

    def __init__(self,
                 batch_size,
                 num_workers,
                 root_dir_web,
                 root_dir_imgnet,
                 num_class=50):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir_web
        self.root_dir_imgnet = root_dir_imgnet

    def run(self):
        train_dataset = webvision_dataset(root_dir=self.root_dir,
                                          mode="train",
                                          num_class=self.num_class)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        test_dataset = webvision_dataset(root_dir=self.root_dir,
                                         mode='test',
                                         num_class=self.num_class)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        imagenet_val = imagenet_dataset(root_dir=self.root_dir_imgnet,
                                        num_class=self.num_class)
        imagenet_loader = DataLoader(dataset=imagenet_val,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)

        return train_loader, test_loader, imagenet_loader


if __name__ == '__main__':
    webvision_data = webvision_dataloader(batch_size=32, num_class=50, num_workers=4, \
        root_dir_web='/home/yfli/webvision/', root_dir_imgnet='/home/yfli/ImageNet1k/')
    webvision_data.run()