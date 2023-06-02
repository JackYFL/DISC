# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : MetaLearning.py (refer to Junnan Li's CVPR 2019 paper "Learning to Learn from Noisy Labeled Data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss
from tqdm import tqdm
import math
import random
from collections import OrderedDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MetaLearning:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']
        self.meta_lr = config['meta_lr']
        self.start_iter = config['start_iter']
        self.epoch_decay_start = config['epoch_decay_start']
        self.mid_iter = config['mid_iter']
        self.num_fast = config['num_fast']
        self.perturb_ratio = config['perturb_ratio']
        self.init = True
        self.eps = config['eps']
        self.alpha = 1

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
        self.net = get_model(config['model1_type'], input_channel, num_classes, device)
        self.tch_net = get_model(config['model1_type'], input_channel, num_classes, device)
        self.pretrain_net = get_model(config['model1_type'], input_channel, num_classes, device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum = 0.9, weight_decay = 1e-3)
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
        self.consistent_criterion = nn.KLDivLoss()
        
    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.tch_net.eval()  # Change model to 'eval' mode

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits = self.tch_net(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()

        acc = 100 * float(correct) / float(total)
        return acc

    def train(self, train_loader, epoch):
        print('Training ...')

        self.net.train()
        
        lr = self.lr
        if self.adjust_lr == True:
            self.adjust_learning_rate(self.optimizer, lr, epoch)
        import ipdb; ipdb.set_trace()
        pbar = tqdm(train_loader)
        for batch_idx, (images, labels) in enumerate(pbar):
            x = Variable(images).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)

            outputs = self.net(x)
            # loss_sup = F.cross_entropy(logits, labels)
            loss_sup = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss_sup.backward(retain_graph = True)
            
            if batch_idx>self.start_iter or epoch > 1:
                if batch_idx > self.mid_iter or epoch>1:
                    self.eps = 0.999
                    alpha = self.alpha
                else:
                    u = (batch_idx-self.start_iter)/(self.mid_iter-self.start_iter)
                    alpha = self.alpha*math.exp(-5*(1-u)**2)    
                    
                if self.init:
                    self.init = False
                    for param,param_tch in zip(self.net.parameters(),self.tch_net.parameters()): 
                        param_tch.data.copy_(param.data)  
                else:
                    for param,param_tch in zip(self.net.parameters(),self.tch_net.parameters()):
                        param_tch.data.mul_(self.eps).add_((1-self.eps), param.data)  
                        
                _,feats = self.pretrain_net(x,get_feat=True)
                tch_outputs = self.tch_net(x,get_feat=False)
                p_tch = F.softmax(tch_outputs,dim=1)
                p_tch = p_tch.detach()

                for i in range(self.num_fast):
                    targets_fast = labels.clone()
                    randidx = torch.randperm(labels.size(0))
                    for n in range(int(labels.size(0)*self.perturb_ratio)):
                        num_neighbor = 10
                        idx = randidx[n]
                        feat = feats[idx]
                        feat.view(1,feat.size(0))
                        feat.data = feat.data.expand(labels.size(0),feat.size(0))
                        dist = torch.sum((feat-feats)**2,dim=1)
                        _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                        targets_fast[idx] = labels[neighbor[random.randint(1,num_neighbor)]]
                        
                    fast_loss = self.criterion(outputs,targets_fast)

                    grads = torch.autograd.grad(fast_loss, self.net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                    for grad in grads:
                        grad = grad.detach()
                        grad.requires_grad = False  
    
                    fast_weights = OrderedDict((name, param - self.meta_lr*grad) for ((name, param), grad) in zip(self.net.named_parameters(), grads))
                    #########load weight########
                    # for k, v in self.net.named_parameters():
                    #     if k in fast_weights:
                    #         v = fast_weights[k]
                    # pre_trained_dict={}
                    # for k, v in self.net.named_parameters():
                    #     if k in fast_weights:
                    #         pre_trained_dict[k] = fast_weights[k].clone().detach()
                    #     else:
                    #         pre_trained_dict[k] = v
                    # model_dict = self.net.state_dict()
                    # model_dict.update(pre_trained_dict)
                    # self.net.load_state_dict(model_dict)
                    ############################
                    
                    fast_out = self.net.forward(x, weights = fast_weights)  
                    logp_fast = F.log_softmax(fast_out,dim=1)
            
                    if i == 0:
                        consistent_loss = self.consistent_criterion(logp_fast,p_tch)
                    else:
                        consistent_loss = consistent_loss + self.consistent_criterion(logp_fast,p_tch)
                
                meta_loss = consistent_loss*alpha/self.num_fast 
                
                meta_loss.backward()
                
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], loss_sup: %.4f'
                    % (epoch + 1, self.epochs, loss_sup.data.item()))
            

    def adjust_learning_rate(self, optimizer, lr, epoch):
        if epoch == self.epoch_decay_start:
            lr = lr/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
