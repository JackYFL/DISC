# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : NL.py (refer to Youngdong Kim's CVPR 2018 paper "NLNL: Negative Learning for Noisy Labels")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NegtiveLearning:
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

        # scratch
        self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
        self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.adjust_lr = config['adjust_lr']
        self.ln_neg = config['ln_neg']

        # loss function
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes)
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes)
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes)
        self.criterion_nll = nn.NLLLoss()

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
            self.adjust_learning_rate(self.optimizer, epoch)
        
        # import ipdb; ipdb.set_trace()
        
        pbar = tqdm(train_loader)
        for (images, labels) in pbar:
            x = Variable(images).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            labels_neg = ((labels.unsqueeze(-1).repeat(1, self.ln_neg))\
                + torch.LongTensor(len(labels), self.ln_neg).random_(1, self.num_classes).to(self.device)) % self.num_classes
            
            assert labels_neg.max() <= self.num_classes-1
            assert labels_neg.min() >= 0
            assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels) * self.ln_neg
            
            logits = self.model_scratch(x)
            ###neg#####
            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))
            neg_num = float((labels_neg>=0).sum())
            loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) 
            ###########
            ###pos#####
            pos_num = float((labels>=0).sum())
            loss_pos = self.criterion(logits, labels)
            ###########
                        
            loss_sup = (loss_neg * neg_num + loss_pos * pos_num)/ (neg_num + pos_num)

            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
