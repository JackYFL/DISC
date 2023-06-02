# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : PENCIL.py (refer to Yi Kun's CVPR 2019 paper "Probabilistic end-to-end noise correction for learning with noisy labels")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss
from tqdm import tqdm
from torch.distributions.beta import Beta

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PENCIL:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']
        self.lr2 = config['lr2']
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
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.stage1 = config['stage1']
        self.stage2 = config['stage2']
        self.lamd = config['lamd']

        # scratch
        self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
        self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, 
                                         momentum=0.9, weight_decay=1e-4)
        if 'cifar' in config['dataset']:
            N = 50000
            self.N = N
        self.new_y = torch.zeros(self.N, self.num_classes).to(self.device)
        
        # loss function
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes)
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes)
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
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

        pbar = tqdm(train_loader)
        for (images, labels, index) in pbar:
            b = len(labels)
            x = Variable(images[0]).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            logits = self.model_scratch(x)
            if epoch < self.stage1:
                lc = self.criterion(logits, labels)
                onehots = torch.zeros(b, self.num_classes).to(self.device).scatter_(1, labels.view(-1, 1), 1)
                self.new_y[index] = onehots
            else:
                yy = self.new_y[index].detach()
                yy.requires_grad_(True)
                #Obtain label distributions (y_hat)
                last_y_var = self.softmax(yy)
                lc = torch.mean(self.softmax(logits)*(self.logsoftmax(logits)-torch.log(last_y_var)))
                #lo is compatibility loss
                lo = self.criterion(last_y_var, labels)
                
            # le is entropy loss
            le = - torch.mean(torch.mul(self.softmax(logits), self.logsoftmax(logits)))

            if epoch < self.stage1:
                loss_sup = lc
            elif epoch < self.stage2:
                loss_sup = lc + self.alpha * lo + self.beta * le
            else:
                loss_sup = lc
                
            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))
            if epoch >= self.stage1 and epoch < self.stage2:
                # import ipdb; ipdb.set_trace()
                yy=yy - self.lamd*yy.grad
                self.new_y[index] = yy.detach()

        self.adjust_learning_rate(self.optimizer, epoch)
                
    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate"""
        if epoch < self.stage2 :
            lr = self.lr
        elif epoch < (self.epochs - self.stage2)//3 + self.stage2:
            lr = self.lr2
        elif epoch < 2 * (self.epochs - self.stage2)//3 + self.stage2:
            lr = self.lr2//10
        else:
            lr = self.lr2//100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
