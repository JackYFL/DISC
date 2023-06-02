# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : GJS.py (refer to Erik Englesson's NeurlPS 2021 paper "Generalized jensen-shannon divergence loss for learning with noisy labels")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import GJSLoss
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GJS:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):
        if config['dataset'] == 'cifar-100':
            self.lr = 0.04
        elif config['dataset'] == 'cifar-10':
            self.lr = 0.1

        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']

        # scratch            
        self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
        self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, nesterov = config['nesterov'],
                                         momentum = config['momentum'], weight_decay = config['weight_decay'])

        # -- Learning Rate Scheduling --
        if config['lr_schedule'] == 'drop':
            lambda_fn = lambda epoch: self.set_drop_lr(epoch, self.lr, self.lr,
                                                config['lr_drop1'], config['lr_drop2'])
        elif config['lr_schedule'] == 'cos':
            lambda_fn = lambda epoch: np.cos(config['cosw'] * np.pi * epoch / config['epochs'])
    
        if config['lr_schedule'] == 'stepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=config['stepLRfactor'])
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_fn)

        # loss function
        if config['dataset'] == 'cifar-10':
            if config['noise_type'] == 'sym':            
                if config['percent'] == 0.2:
                    js_weights = '0.3 0.35 0.35'
                elif config['percent'] == 0.5:
                    js_weights = '0.1 0.45 0.45'
                elif config['percent'] == 0.8:
                    js_weights = '0.1 0.45 0.45'
            elif config['noise_type'] == 'asym':
                if config['percent'] == 0.4:
                    js_weights = '0.3 0.35 0.35'
                    
        elif config['dataset'] == 'cifar-100':
            if config['noise_type'] == 'sym':            
                if config['percent'] == 0.2:
                    js_weights = '0.3 0.35 0.35'
                elif config['percent'] == 0.5:
                    js_weights = '0.9 0.05 0.05'
                elif config['percent'] == 0.8:
                    js_weights = '0.1 0.45 0.45'
            elif config['noise_type'] == 'asym':
                if config['percent'] == 0.4:
                    js_weights = '0.1 0.45 0.45'
                
        self.criterion = GJSLoss(num_classes, weights = js_weights)

    def set_drop_lr(self, epoch, lr_warmup, lr_start, drop1=40, drop2=80, start_epoch=0, drop_factor=0.1):
        assert drop1 < drop2

        if epoch > start_epoch and epoch < drop1:
            return lr_start / lr_warmup
        elif epoch >= drop1 and epoch < drop2:
            return drop_factor * (lr_start / lr_warmup)
        elif epoch >= drop2:
            return drop_factor * drop_factor * (lr_start / lr_warmup)
        else:
            return 1.0
        
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
        for (images, labels, _) in pbar:
            x = torch.cat(images[:2], 0).to(self.device)
            labels = Variable(labels).to(self.device)

            logits = self.model_scratch(x)
            logits = list(torch.split(logits, images[0].size(0)))
            # loss_sup = F.cross_entropy(logits, labels)
            loss_sup = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))

        self.scheduler.step()
