# Thanks to https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py

from turtle import forward
import torch
import torch.nn.functional as F

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
  
class SCELoss(torch.nn.Module):
    def __init__(self, dataset, num_classes=10, gpu=None):
        super(SCELoss, self).__init__()
        self.device = torch.device('cuda:%s'%gpu) if gpu else torch.device('cpu')
        if dataset == 'cifar-10':
            self.alpha, self.beta = 0.1, 1.0
        elif dataset == 'cifar-100':
            self.alpha, self.beta = 6.0, 0.1
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class GCELoss(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, gpu=None):
        super(GCELoss, self).__init__()
        self.device = torch.device('cuda:%s'%gpu) if gpu else torch.device('cpu')
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

class DMILoss(torch.nn.Module):
    def __init__(self, num_classes, gpu=None):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device('cuda:%s'%gpu) if gpu else torch.device('cpu')

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).to(self.device)
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

class pNorm(torch.nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

class CE_SR(torch.nn.Module):
    def __init__(self, dataset):
        super(CE_SR, self).__init__()
        if dataset == 'cifar-10':
            p = 0.1
            self.tau = 0.5
            self.lamd = 1.2
        elif dataset == 'cifar-100':
            p = 0.01
            self.tau = 0.5
            self.lamd = 1
        self.norm = pNorm(p)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        loss = self.criterion(pred / self.tau, target) + self.lamd * self.norm(pred / self.tau)
        return loss
