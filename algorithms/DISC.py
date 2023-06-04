# -*- coding: utf-8 -*-
# @Author : Jack
# @Email  : liyifan20g@ict.ac.cn
# @File   : DISC.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50, vgg19_bn

from models import InceptionResNetV2
import numpy as np
from utils import get_model
from losses import GCELoss, Mixup
from tqdm import tqdm
import pickle
import os

class DISC:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):
        self.num_classes = num_classes
        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.epochs = config['epochs']
        self.tmp_epoch = 0
        self.dataset = config['dataset']
        self.noise_type = config['noise_type']+'_'+str(config['percent'])
        
        # Backbones for different datasets
        if 'cifar' in self.dataset or 'tiny_imagenet' in self.dataset:
            if 'ins' in config['noise_type']:
                config['model1_type'] = 'resnet34'
            self.model_scratch = get_model(config['model1_type'], input_channel, num_classes, device)
            
        elif 'clothing' in self.dataset or 'food' in self.dataset:    
            self.model_scratch = resnet50(pretrained=True)
            self.model_scratch.fc = nn.Linear(2048, config['num_classes'])
            self.model_scratch = self.model_scratch.to(self.device)
            
        elif 'animal' in self.dataset:    
            self.model_scratch = vgg19_bn(pretrained=False)
            self.model_scratch.classifier._modules['6'] = nn.Linear(4096, 10)
            self.model_scratch = self.model_scratch.to(self.device)

        elif 'webvision' in self.dataset:    
            self.model_scratch = InceptionResNetV2(config['num_classes'])
            self.model_scratch.fc = nn.Linear(2048, config['num_classes'])
            self.model_scratch = self.model_scratch.to(self.device)

        # Thresholds for different subsets
        self.adjust_lr = 1
        self.lamd_ce = 1
        self.lamd_h = 1
        self.sigma = 0.5
        self.momentum = config['momentum']

        if config['dataset'] == 'cifar-10':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 20
            
            if config['noise_type']=='ins':
                self.start_epoch = 15
                
            if config['percent'] in [0.6, 0.8]:
                self.lamd_h = 0.2
                self.momentum = 0.95

        elif config['dataset'] == 'cifar-100':
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 20
            
            if config['noise_type']=='ins':
                self.start_epoch = 15
            
            if config['percent'] == 0.6:
                self.lamd_h = 0.2
                self.momentum = 0.95
            
            elif config['percent'] == 0.8:
                self.start_epoch = 20
                self.lamd_h = 0.2 #0.5
                self.momentum = 0.95

        elif config['dataset'] == 'tiny_imagenet':
            self.momentum = 0.99
            if config['noise_type']=='asym':
                self.momentum = 0.95
            self.start_epoch = 15
            
        elif config['dataset'] == 'clothing1M':
            self.start_epoch = 10
            self.momentum = 0.95
            
        elif config['dataset'] == 'food101N':
            self.momentum = 0.95
            self.start_epoch = 10

        elif config['dataset'] == 'animal10N':
            self.momentum = 0.99
            self.start_epoch = 10

        elif config['dataset'] == 'webvision':
            self.momentum = 0.99
            self.lamd_h = 1 #TODO 
            self.start_epoch = 15
            
        config['start_epoch'] = self.start_epoch
        config['momentum'] = self.momentum

        # Optimizers for different subsets
        if config['percent'] > 0.6 and 'cifar' in config['dataset']:
            self.lr = 0.001
            # Adjust learning rate and betas for Adam Optimizer
            mom1 = 0.9
            mom2 = 0.1
            self.alpha_plan = [self.lr] * config['epochs']
            self.beta1_plan = [mom1] * config['epochs']
            self.epoch_decay_start = config['epoch_decay_start']
            for i in range(config['epoch_decay_start'], config['epochs']):
                self.alpha_plan[i] = float(config['epochs'] - i) / (config['epochs'] - config['epoch_decay_start']) * self.lr
                self.beta1_plan[i] = mom2

            self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
            config['optimizer'] = 'adam' 
            self.optim_type = 'adam'

        else:
            if 'cifar' in config['dataset']:
                self.lr = 0.1
                self.weight_decay = 1e-3
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 160], verbose=True)

                config['weight_decay'] = self.weight_decay

            elif 'tiny_imagenet' in config['dataset']:
                self.lr = 0.1
                self.weight_decay = 1e-3
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 160], verbose=True)

                config['weight_decay'] = self.weight_decay

            elif 'clothing' in config['dataset']:
                self.lr = 0.01

                self.weight_decay = 5e-4
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 80], verbose=True)

            elif 'food' in config['dataset']:
                self.lr = 0.01

                self.weight_decay = 5e-4
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,80], verbose=True)

            elif 'animal' in config['dataset']:
                self.lr = 0.05

                self.weight_decay = 5e-4
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[50,80], verbose=True)
                                                   
            elif 'webvision' in config['dataset']:
                self.lr = 0.1 
                self.weight_decay = 5e-4 
                self.optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 80], verbose=True) 

                config['optimizer'] = 'sgd' 
                self.optim_type = 'sgd'

        config['lr'] = self.lr
        
        if 'cifar' in config['dataset']:
            N = 50000
        elif 'clothing1M' in config['dataset']:
            N = 1037498
        elif 'tiny_imagenet' in config['dataset']:
            N = 100000
        elif 'food' in config['dataset']:
            N = 75750
        elif 'animal' in config['dataset']:
            N = 50000
        elif 'webvision' in config['dataset']:
            N = 65944
        self.N = N
        
        # Variable definition
        self.s_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.w_prev_confidence = torch.ones(N).to(self.device)*1/N
        self.ws_prev_confidence = torch.ones(N).to(self.device)*1/N

        self.w_probs = torch.zeros(N, config['num_classes']).to(self.device)
        self.s_probs = torch.zeros(N, config['num_classes']).to(self.device)
        self.labels = torch.ones(N).long().to(self.device)
        if 'cifar' in config['dataset']:
            self.gt_labels = torch.tensor(self.get_gt_labels(config['dataset'], config['root'])).to(self.device)
        self.weak_labels = self.labels.detach().clone()

        self.clean_flags = torch.zeros(N).bool().to(self.device)
        self.hard_flags = torch.zeros(N).bool().to(self.device)
        self.correction_flags = torch.zeros(N).bool().to(self.device)
        self.weak_flags = torch.zeros(N).bool().to(self.device)
        self.w_selected_flags = torch.zeros(N).bool().to(self.device)
        self.s_selected_flags = torch.zeros(N).bool().to(self.device)
        self.selected_flags = torch.zeros(N).bool().to(self.device)
        self.class_weight = torch.ones(self.num_classes).to(self.device)
        self.accs = np.zeros(self.epochs)
        self.acc_list = list()
        self.num_list = list()
        
        # Loss function definition
        self.GCE_loss = GCELoss(num_classes=num_classes, gpu=config['gpu'])
        self.mixup_loss = Mixup(gpu=config['gpu'], num_classes=num_classes, alpha=config['alpha'])
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model_scratch.train()
        self.tmp_epoch = epoch
        pbar = tqdm(train_loader)
        if epoch < self.start_epoch:
            for (images, targets, indexes) in pbar:
                w_imgs, s_imgs = Variable(images[0]).to(self.device, non_blocking=True), \
                                Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)

                all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
                bs = w_imgs.shape[0]
                logits = self.model_scratch(all_inputs)
                w_logits = logits[:bs]
                s_logits = logits[bs:]
                loss_sup = self.criterion(w_logits, targets) \
                         + self.criterion(s_logits, targets)
                self.optimizer.zero_grad()
                loss_sup.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    w_prob = F.softmax(w_logits, dim=1)
                    self.w_probs[indexes] = w_prob
                    s_prob = F.softmax(s_logits, dim=1)
                    self.s_probs[indexes] = s_prob
                    
                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))
        else:
            for (images, targets, indexes) in pbar:
                w_imgs, s_imgs = Variable(images[0]).to(self.device, non_blocking=True), \
                                 Variable(images[1]).to(self.device, non_blocking=True)
                targets = Variable(targets).to(self.device)
                
                #############CE+GCE############
                all_inputs = torch.cat([w_imgs, s_imgs], dim=0)
                bs = w_imgs.shape[0]
                logits = self.model_scratch(all_inputs)
                w_logits = logits[:bs]
                s_logits = logits[bs:]
                
                loss_sup = torch.tensor(0).float().to(self.device)
                
                b_clean_flags = self.clean_flags[indexes]
                clean_num = b_clean_flags.sum()
                b_hard_flags = self.hard_flags[indexes]
                hard_num = b_hard_flags.sum()
                
                batch_size = len(w_imgs)

                if clean_num:
                    clean_loss_sup = self.criterion(w_logits[b_clean_flags], targets[b_clean_flags]) \
                            + self.criterion(s_logits[b_clean_flags], targets[b_clean_flags])
                    loss_sup += clean_loss_sup * self.lamd_ce * (clean_num/batch_size) 
                if hard_num:
                    hard_loss_sup = self.GCE_loss(w_logits[b_hard_flags], targets[b_hard_flags]) \
                        + self.GCE_loss(s_logits[b_hard_flags], targets[b_hard_flags])
                    loss_sup += hard_loss_sup * self.lamd_h * (hard_num/batch_size)
                ###########################
                
                ############Mixup##########
                weak_labels = self.weak_labels[indexes]      
                weak_flag = self.weak_flags[indexes] 
                weak_num = weak_flag.sum() 

                if weak_num:        
                    mixup_loss = self.mixup_loss(w_imgs[weak_flag], weak_labels[weak_flag], self.model_scratch)
                    mixup_loss += self.mixup_loss(s_imgs[weak_flag], weak_labels[weak_flag], self.model_scratch)
                    loss_sup += mixup_loss
                    
                #######################
                with torch.no_grad():
                    w_prob = F.softmax(w_logits, dim=1)
                    self.w_probs[indexes] = w_prob
                    s_prob = F.softmax(s_logits, dim=1)
                    self.s_probs[indexes] = s_prob

                if loss_sup:
                    self.optimizer.zero_grad()
                    loss_sup.backward()
                    self.optimizer.step()

                pbar.set_description(
                        'Epoch [%d/%d], loss_sup: %.4f'
                        % (epoch + 1, self.epochs, loss_sup.data.item()))

        with torch.no_grad():
            ws_probs = (self.w_probs+self.s_probs)/2
            w_prob_max, w_label= torch.max(self.w_probs, dim=1)
            s_prob_max, s_label= torch.max(self.s_probs, dim=1)
            ws_prob_max, ws_label= torch.max(ws_probs, dim=1)
            labels = w_label.clone()
            
            ###############Selection###############
            w_mask = self.w_probs[self.labels>=0, self.labels]>self.w_prev_confidence[self.labels>=0]
            s_mask = self.s_probs[self.labels>=0, self.labels]>self.s_prev_confidence[self.labels>=0]
            self.clean_flags = w_mask & s_mask
            self.selected_flags = w_mask + s_mask
            self.w_selected_flags = w_mask & (~self.clean_flags) #H_w
            self.s_selected_flags = s_mask & (~self.clean_flags) #H_s
            self.hard_flags = self.w_selected_flags + self.s_selected_flags #H       
            #######################################

            ###############Correction##############
            ws_threshold = (self.w_prev_confidence + self.s_prev_confidence)/2 + self.sigma
            ws_threshold = torch.min(ws_threshold, torch.tensor(0.90).to(self.device))
            self.correction_flags = ws_prob_max > ws_threshold
            self.correction_flags = self.correction_flags & (~ self.selected_flags) # P-(C+H)
            #######################################
            
            ###############Mix set###############
            self.weak_flags = self.correction_flags + self.selected_flags
            self.weak_labels[self.selected_flags] = self.labels[self.selected_flags]
            self.weak_labels[self.correction_flags] = ws_label[self.correction_flags]
            #######################################

            self.w_prev_confidence = self.momentum*self.w_prev_confidence + (1-self.momentum)*w_prob_max
            
            self.s_prev_confidence = self.momentum*self.s_prev_confidence + (1-self.momentum)*s_prob_max
            
            if 'cifar' in self.dataset:
                clean_acc = (self.labels[self.clean_flags]==self.gt_labels[self.clean_flags]).sum()/self.clean_flags.sum()
                hard_acc = (self.labels[self.hard_flags]==self.gt_labels[self.hard_flags]).sum()/self.hard_flags.sum()
                selection_acc = (self.labels[self.selected_flags]==self.gt_labels[self.selected_flags]).sum()/self.selected_flags.sum()
                w_selection_acc = (self.labels[self.w_selected_flags]==self.gt_labels[self.w_selected_flags]).sum()/self.w_selected_flags.sum()
                s_selection_acc = (self.labels[self.s_selected_flags]==self.gt_labels[self.s_selected_flags]).sum()/self.s_selected_flags.sum()
                correction_acc = (self.weak_labels[self.correction_flags]==self.gt_labels[self.correction_flags]).sum()/self.correction_flags.sum()
                weak_acc = (self.weak_labels[self.weak_flags]==self.gt_labels[self.weak_flags]).sum()/self.weak_flags.sum()
                total_acc = (labels == self.gt_labels).sum()/self.N
                print("Clean ratio is %.4f, clean num is %d"%(clean_acc, self.clean_flags.sum()))
                print("Hard ratio is %.4f, hard num is %d"%(hard_acc, self.hard_flags.sum()))
                print("Weak selection ratio is %.4f, weak selection num is %d"%(w_selection_acc, self.w_selected_flags.sum()))
                print("Strong selection ratio is %.4f, strong selection num is %d"%(s_selection_acc, self.s_selected_flags.sum()))
                print("Selection ratio is %.4f, selection num is %d"%(selection_acc, self.selected_flags.sum()))
                print("Correction ratio is %.4f, correction num is %d"%(correction_acc, self.correction_flags.sum()))
                print("Weak ratio is %.4f, weak num is %d"%(weak_acc, self.weak_flags.sum()))
                print("Total ratio is %.4f"%(total_acc))
                self.acc_list.append((clean_acc.cpu().numpy(), hard_acc.cpu().numpy(), w_selection_acc.cpu().numpy(), s_selection_acc.cpu().numpy(), selection_acc.cpu().numpy(), correction_acc.cpu().numpy(), weak_acc.cpu().numpy(), total_acc.cpu().numpy()))
            else:
                clean_num = self.clean_flags.sum()
                hard_num = self.hard_flags.sum()
                w_selection_num = self.w_selected_flags.sum()
                s_selection_num = self.s_selected_flags.sum()
                selection_num = self.selected_flags.sum()
                correction_num = self.correction_flags.sum()
                weak_num = self.weak_flags.sum()
                print("Clean num is %d"%clean_num)
                print("Hard num is %d"%hard_num)
                print("Weak selection num is %d" % w_selection_num)
                print("Strong selection num is %d" % s_selection_num)
                print("selection num is %d"%selection_num)
                print("Correction num is %d"%correction_num)
                print("Weak num is %d"%weak_num)
        self.num_list.append((self.clean_flags.sum().cpu().numpy(), self.hard_flags.sum().cpu().numpy(), self.w_selected_flags.sum(), self.s_selected_flags.sum(), self.correction_flags.sum().cpu().numpy(), self.weak_flags.sum().cpu().numpy()))

        if epoch==(self.epochs-1):
            self.save_results()
        if self.adjust_lr:
            if self.optim_type == 'sgd':
                self.scheduler.step()
            elif self.optim_type == 'adam':
                self.adjust_learning_rate(self.optimizer, epoch)
                print("lr is %.8f." % (self.alpha_plan[epoch]))
                
    def evaluate(self, test_loader):
        print('Evaluating ...')

        self.model_scratch.eval()  # Change model to 'eval' mode

        correct = 0
        correct_top5 = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).to(self.device)
            logits = self.model_scratch(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            if 'webvision' in self.dataset:
                _, index = outputs.topk(5)
                index = index.cpu()
                correct_tmp = index.eq(labels.view(-1, 1).expand_as(index))
                correct_top5 += correct_tmp.sum().cpu()
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
        
        acc = 100 * float(correct) / float(total)
        self.accs[self.tmp_epoch] = acc
        if 'webvision' in self.dataset:
            acc_top5 =100 * float(correct_top5)/float(total)
            return acc, acc_top5

        if 'clothing1M' in self.dataset and self.tmp_epoch>1:
            checkpoint_root = 'checkpoints/%s/'%self.dataset
            if acc > max(self.accs[:self.tmp_epoch]):
                print('| Saving Best Net%d ...'%self.tmp_epoch)
                save_point = checkpoint_root+'DSCO_prevthresh.pth.tar'
                torch.save(self.model_scratch.state_dict(), save_point)

        return acc

    def save_checkpoints(self):
        checkpoint_root = 'checkpoints/%s/'%self.dataset
        filename = checkpoint_root + 'save_epoch199_%s'%self.noise_type
        
        if not os.path.exists(checkpoint_root):
            os.makedirs(checkpoint_root)

        state = {'weak_labels':self.weak_labels, 'weak_flags':self.weak_flags, 'weak selected flags':self.w_selected_flags, 
                 'strong selected flags':self.s_selected_flags, 'clean_flags':self.clean_flags, 'labels':self.labels,
                 'hard_flags':self.hard_flags, 'correction_flags':self.correction_flags, 'w_probs':self.w_probs, 's_probs':self.s_probs, 
                 'epoch': self.start_epoch,'model': self.model_scratch.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename+'.pth')
        print("The model has been saved !!!!!")
    
    def save_results(self, name='disc'):
        save_root = 'result_root/%s/'%self.dataset
        filename = save_root + self.noise_type + '_save_' + name + '.npy'

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if 'cifar' in self.dataset:
            results = {'num_list': self.num_list, 'acc_list': self.acc_list, 'test_acc': self.accs}
        else:
            results = {'num_list': self.num_list, 'test_acc': self.accs}
        np.save(filename, results)

    def load_checkpoints(self):
        filename = 'checkpoints/%s/start_epoch%d.pth'%(self.dataset, self.start_epoch)
        model_parameters = torch.load(filename)
        self.model_scratch.load_state_dict(model_parameters['model'])
        self.s_selected_flags = model_parameters['strong selected flags']      
        self.w_selected_flags = model_parameters['weak selected flags']      
        self.weak_labels = model_parameters['weak_labels'].to(self.device)
        self.weak_flags = model_parameters['weak_flags'].to(self.device)
        self.clean_flags = model_parameters['clean_flags'].to(self.device)
        self.correction_flags = model_parameters['correction_flags'].to(self.device)
        self.hard_flags = model_parameters['hard_flags'].to(self.device)
        self.w_probs = model_parameters['w_probs'].to(self.device)
        self.s_probs = model_parameters['s_probs'].to(self.device)
        print("The model has been loaded !!!!!")

    def get_clothing_labels(self, root):
        # import ipdb; ipdb.set_trace()
        with open('%s/label/noisy_label_kv.txt'%root,'r') as f:
            lines = f.read().splitlines()
            train_labels = {}
            for l in lines:
                entry = l.split(' ')
                img_path = entry[0]
                if os.path.exists(root+'/'+img_path):                        
                    # self.train_imgs.append(img_path)                                               
                    train_labels[img_path] = int(entry[1])

        with open('%s/label/noisy_train_key_list.txt'%root,'r') as f:
            lines = f.read().splitlines()
            for i, l in enumerate(lines):
                img_path = l.strip()
                self.labels[i]=torch.tensor(train_labels[img_path]).long()

    def get_food101N_labels(self, root):
        # import ipdb; ipdb.set_trace()
        class2idx = {}
        imgs_root = root + '/images/'
        train_file = root + '/train.txt'
        imgs_dirs = os.listdir(imgs_root)
        imgs_dirs.sort()
        
        for idx, food in enumerate(imgs_dirs):
            class2idx[food] = idx
            
        with open(train_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                train_img = line.strip()
                target = class2idx[train_img.split('/')[0]]
                self.labels[i]=torch.tensor(int(target)).long()

    def get_animal10N_labels(self, root):
        imgs_root = root + '/training/'
        imgs_dirs = os.listdir(imgs_root)
        
        for i, img in enumerate(imgs_dirs):
            self.labels[i] = torch.tensor(int(img[0])).long()

    def get_webvision_labels(self, root):
        with open(root+'info/train_filelist_google.txt') as f:
            lines=f.readlines()    
            i=0
            for line in lines:
                _, target = line.split()
                target = int(target)
                if target<50:
                    self.labels[i]=torch.tensor(target).long()
                    i+=1

    def get_labels(self, train_loader):
        print("Loading labels......")
        pbar = tqdm(train_loader)
        for (_, targets, indexes) in pbar:
            targets = targets.to(self.device)
            self.labels[indexes] = targets
        print("The labels are loaded!")

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

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
