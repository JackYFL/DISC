# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : ELR.py (refer to Sheng Liu's NeurlPS 2020 paper "Early-learning regularization prevents memorization of noisy labels")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import get_model
from losses import elr_loss
from tqdm import tqdm
from torchvision.models import resnet50, vgg19_bn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ELR:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']

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
        self.adjust_lr = config['adjust_lr']
        N=0

        # scratch
        if 'cifar-10' in self.dataset:
            lr = 0.02
            wd = 1e-3
            if config['noise_type']=='asym':
                beta = 0.9
                lamd = 1
                self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)            
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=lr, weight_decay=wd)
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80], gamma=0.01, verbose=True)
            elif config['noise_type']=='sym':
                beta = 0.7
                lamd = 3
                self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)            
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=lr, weight_decay=wd)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=0.001, verbose=True)
            N = 50000
        
        elif 'cifar-100' in self.dataset:
            lr = 0.02
            wd = 1e-3
            beta = 0.9
            lamd = 7
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)            
            self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=lr, weight_decay=wd)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 120], gamma=0.01, verbose=True)
            N = 50000
            
        elif 'clothing' in self.dataset:    
            import ipdb; ipdb.set_trace()
            self.model_scratch = resnet50(pretrained=True)
            # import ipdb; ipdb.set_trace()
            self.model_scratch.fc = nn.Linear(2048, config['num_classes'])
            self.model_scratch = self.model_scratch.to(self.device)
            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            self.adjust_lr = config['adjust_lr']

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

        # loss function
        self.criterion = elr_loss(num_examp=N, device=self.device, num_classes=num_classes, beta=beta, lamd=lamd)
        
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
        
        if epoch <= self.warmup_epoch:                                
            pbar = tqdm(train_loader)
            for (images, labels, indexes) in pbar:
                x = Variable(images[0]).to(self.device, non_blocking=True)
                labels = Variable(labels).to(self.device)

                logits = self.model_scratch(x)
                loss_sup = self.criterion(indexes, logits, labels)

                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
        else:
            self.model_scratch.train()
            pbar = tqdm(train_loader)
            for (images, targets, indexes) in pbar:
                w_imgs, _ = Variable(images[0]).to(self.device, non_blocking=True), \
                                Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)
                
                logits = self.model_scratch(w_imgs)
                self.criterion.update_hist(epoch, logits.data.detach(), indexes.numpy().tolist())

                loss = self.cross_entropy(logits, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                pbar.set_description(
                'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss.data.item()))
            
        self.scheduler.step()
                
    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
