# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : Mixup.py (refer to Hongyi Zhang's ICLR 2018 paper "Mixup: Beyond Empirical risk minimization")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss
from torchvision.models import resnet50, vgg19_bn
from tqdm import tqdm
from torch.distributions.beta import Beta

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Mixup:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']
        self.num_classes = num_classes
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']
        self.dataset = config['dataset']

        # scratch
        if 'cifar' in self.dataset:
            # scratch
            if 'ins' in config['noise_type']:
                config['model1_type'] = 'resnet34'
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'

        elif 'animal' in self.dataset:
            self.model_scratch = vgg19_bn(pretrained=False)
            self.model_scratch.classifier._modules['6'] = nn.Linear(4096, 10)
            self.model_scratch = self.model_scratch.to(self.device)
            self.lr = 0.05
            self.weight_decay = 5e-4
            self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[50,80], verbose=True)
            config['optimizer'] = 'sgd' 
            self.optim_type = 'sgd'
            config['weight_decay'] = self.weight_decay

        elif 'food' in config['dataset']:
            self.model_scratch = resnet50(pretrained=True)
            self.model_scratch.fc = nn.Linear(2048, config['num_classes'])
            self.model_scratch = self.model_scratch.to(self.device)
            self.lr = 0.01
            self.weight_decay = 5e-4
            self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            config['optimizer'] = 'sgd' 
            self.optim_type = 'sgd'
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,80], verbose=True)
            config['weight_decay'] = self.weight_decay

        config['lr'] = self.lr
        self.adjust_lr = config['adjust_lr']

        # loss function
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes)
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes)
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes)

    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits = self.model_scratch(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()

        acc = 100 * float(correct) / float(total)
        return acc

    def train(self, train_loader, epoch):
        print('Training ...')

        self.model_scratch.train()

        if self.adjust_lr == True:
            if self.optim_type=='adam':
                self.adjust_learning_rate(self.optimizer, epoch)
            elif self.optim_type=='sgd':
                self.scheduler.step()

        pbar = tqdm(train_loader)
        for (images, labels) in pbar:
            b = len(labels)
            x = Variable(images).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            onehots = torch.zeros(b, self.num_classes).to(self.device).scatter_(1, labels.view(-1, 1), 1)
            mixed_x, mixed_y = self.org_mixup_data(x, onehots)
            mixed_p = self.model_scratch(mixed_x)
            loss_sup = -torch.mean(torch.sum(F.log_softmax(mixed_p, dim=1) * mixed_y,dim=1))

            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))

    def org_mixup_data(self, x, y, alpha=5.0):
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample() if alpha > 0 else 1
        index = torch.randperm(x.size()[0]).cuda() 
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
