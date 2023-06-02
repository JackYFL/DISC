# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : StandardCE.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model
from losses import SCELoss, GCELoss, DMILoss, CE_SR
from tqdm import tqdm
from torchvision.models import resnet50, vgg19_bn
import os
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class StandardCETest:
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
        self.noise_type = config['noise_type']+'_'+str(config['percent'])
        # scratch
        if 'cifar' in self.dataset:
            if 'ins' in config['noise_type']:
                config['model1_type'] = 'resnet34'
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)            
            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'
            self.gt_labels = torch.tensor(self.get_gt_labels(config['dataset'], config['root'])).to(self.device)
            
        elif 'clothing' in self.dataset:    
            self.model_scratch = resnet50(pretrained=True)
            # import ipdb; ipdb.set_trace()
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'
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
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            print("The loss is ce")
        elif config['loss_type'] == 'ce-sr':
            self.criterion = CE_SR(dataset=config['dataset'])
            print("The loss is ce-sr")
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes, gpu=config['gpu'])
            print("The loss is sce")
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes, gpu=config['gpu'])
            print("The loss is gce")
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes, gpu=config['gpu'])
            print("The loss is dmi")
        self.train_acc_list = []
        self.test_acc_list = []

        if 'cifar' in config['dataset']:
            N = 50000
            self.N = N
        elif 'clothing1M' in config['dataset']:
            N = 1037498
        elif 'food' in config['dataset']:
            N = 75750
        elif 'animal' in config['dataset']:
            N = 50000
        elif 'webvision' in config['dataset']:
            N = 65944
        self.probs = torch.zeros(N, num_classes).to(self.device)
        self.labels = torch.zeros(N).long().to(self.device)

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
        self.test_acc_list.append(acc)
        return acc

    def save_results(self, name='StandardCE'):
        save_root = 'result_root/%s/%s/'%(self.dataset, name)
        filename = save_root + self.noise_type + '.npy'

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if 'cifar' in self.dataset:
            results = {'train_acc': self.train_acc_list, 'test_acc': self.test_acc_list}
            np.save(filename, results)

    def train(self, train_loader, epoch):
        print('Training ...')

        self.model_scratch.train()

        if self.adjust_lr == True:
            if self.optim_type=='adam':
                self.adjust_learning_rate(self.optimizer, epoch)
            elif self.optim_type=='sgd':
                self.scheduler.step()
                                
        if 'clothing' in self.dataset:
            pbar = tqdm(train_loader)
            for (images, targets, _) in pbar:
                w_imgs, _ = Variable(images[0]).to(self.device, non_blocking=True), \
                                Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)
                
                logits = self.model_scratch(w_imgs)
                # loss_sup = F.cross_entropy(logits, labels)
                loss_sup = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
        else:
            pbar = tqdm(train_loader)
            for (images, labels, indexes) in pbar:
                x, _ = Variable(images[0]).to(self.device, non_blocking=True), \
                                 Variable(images[1]).to(self.device, non_blocking=True)
                labels = Variable(labels).to(self.device)

                logits = self.model_scratch(x)
                probs = F.softmax(logits, dim=1)
                self.probs[indexes] = probs
                # loss_sup = F.cross_entropy(logits, labels)
                loss_sup = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
            _, pre_labels = torch.max(self.probs, dim=1)
            train_acc = (pre_labels==self.labels).sum()/self.N
            self.train_acc_list.append(100*float(train_acc.cpu().numpy()))
            print("Training acc is %.4f" % train_acc)
            if epoch==(self.epochs-1):
                self.save_results()

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

    def get_gt_labels(self, dataset, root):
        if dataset=='cifar-10':
            train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
            base_folder = 'cifar-10-batches-py'
        elif dataset=='cifar-100':
            train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            base_folder = 'cifar-100-python'
        targets = []
        for file_name, checksum in train_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        return targets

    def get_labels(self, train_loader):
        # import ipdb; ipdb.set_trace()
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for (_, targets, indexes) in pbar:
            targets = targets.to(self.device)
            self.labels[indexes] = targets
        print("The labels are loaded!")

class StandardCE:
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

        # scratch
        if 'cifar' in self.dataset:
            if 'ins' in config['noise_type']:
                config['model1_type'] = 'resnet34'
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)            
            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'
            
        elif 'clothing' in self.dataset:    
            self.model_scratch = resnet50(pretrained=True)
            # import ipdb; ipdb.set_trace()
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'
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
        if config['loss_type'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            print("The loss is ce")
        elif config['loss_type'] == 'ce-sr':
            self.criterion = CE_SR(dataset=config['dataset'])
            print("The loss is ce-sr")
        elif config['loss_type'] == 'sce':
            self.criterion = SCELoss(dataset=config['dataset'], num_classes=num_classes, gpu=config['gpu'])
            print("The loss is sce")
        elif config['loss_type'] == 'gce':
            self.criterion = GCELoss(num_classes=num_classes, gpu=config['gpu'])
            print("The loss is gce")
        elif config['loss_type'] == 'dmi':
            self.criterion = DMILoss(num_classes=num_classes, gpu=config['gpu'])
            print("The loss is dmi")
        self.save_epochs = []

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
                                
        if 'clothing' in self.dataset:
            pbar = tqdm(train_loader)
            for (images, targets, _) in pbar:
                w_imgs, _ = Variable(images[0]).to(self.device, non_blocking=True), \
                                Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)
                
                logits = self.model_scratch(w_imgs)
                # loss_sup = F.cross_entropy(logits, labels)
                loss_sup = self.criterion(logits, targets)

                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
        else:
            pbar = tqdm(train_loader)
            for (images, labels, indexes) in pbar:
                x = Variable(images).to(self.device, non_blocking=True)
                labels = Variable(labels).to(self.device)

                logits = self.model_scratch(x)
                loss_sup = self.criterion(logits, labels)

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
