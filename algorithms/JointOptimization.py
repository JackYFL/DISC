# -*- coding: utf-8 -*-
# @Author : Jack 
# @Email  : liyifan20g@ict.ac.cn
# @File   : JointOptimization.py (refer to Tanaka's CVPR 2018 paper "Joint optimization framework for learning with noisy labels")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pickle
import os
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class JointOptimization:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']
        self.retrain_lr = config['retrain_lr']

        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']
        self.retrain_epochs = config['retrain_epochs']
        self.num_classes = num_classes
        
        # scratch
        self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
        # self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.adjust_lr = config['adjust_lr']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.begin = config['begin']
        self.count = 0
        
        if 'cifar' in config['dataset']:
            N = 50000
            self.N = N
        self.soft_labels = torch.zeros(N, self.num_classes).to(self.device)
        self.predictions = torch.zeros(N, self.num_classes, 10).to(self.device)
        
        # loss function
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes)
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes)
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes)
            
    def train_loss(self, outputs, soft_targets):
        p = torch.ones(self.num_classes).to(self.device) / self.num_classes
        probs = F.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        
        L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
        L_p = -torch.sum(torch.log(avg_probs) * p)
        L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))
        
        loss = L_c + self.alpha * L_p + self.beta * L_e
        return probs, loss
    
    def retrain_loss(self, outputs, soft_targets):
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
        return loss
    
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

    def label_update(self, results):
        self.count += 1
        
        # While updating the noisy label y_i by the probability, we used the average output probability of
        # the past 10 epochs as s
        with torch.no_grad():
            idx = (self.count-1) % 10
            self.predictions[:, :, idx] = results
            
            if self.count >= self.begin:
                self.soft_labels = self.predictions.mean(axis=-1)
                self.hard_labels = torch.argmax(self.soft_labels, axis=1)
            
    def retrain(self, train_loader, epoch):
        pbar = tqdm(train_loader)
        self.adjust_learning_rate(epoch)
        # import ipdb; ipdb.set_trace()
        for (images, labels, indexes) in pbar:
            x = Variable(images[0]).to(self.device, non_blocking = True)
            labels = Variable(labels).to(self.device)
            
            logits = self.model_scratch(x)
            soft_labels = self.hard_labels[indexes]
            loss_sup = self.retrain_loss(logits, soft_labels)

            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.retrain_epochs, loss_sup.data.item()))

    def train(self, train_loader, epoch):
        print('Training ...')
        # import ipdb; ipdb.set_trace()
        self.model_scratch.train()

        results = torch.zeros(self.N, self.num_classes).to(self.device)
        pbar = tqdm(train_loader)
        for (images, labels, indexes) in pbar:
            x = Variable(images[0]).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            logits = self.model_scratch(x)
            soft_labels = self.soft_labels[indexes]
            probs, loss_sup = self.train_loss(logits, soft_labels)

            results[indexes] = probs.detach()
            self.optimizer.zero_grad()
            loss_sup.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))
        
        self.label_update(results)
        
    def adjust_learning_rate(self, epoch):
        if epoch in [int(self.retrain_epochs / 3), int(self.retrain_epochs * 2 / 3)]:
            self.retrain_lr *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.retrain_lr
                
    def get_labels(self, train_loader):
        # import ipdb; ipdb.set_trace()
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for (_, targets, indexes) in pbar:
            targets = targets.to(self.device)
            self.soft_labels[indexes] = self.soft_labels[indexes].scatter_(1, targets.view(-1, 1), 1)
        print("The soft labels are loaded!")
 