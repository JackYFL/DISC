import torch
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_criterion(num_classes, args):
    alpha = args.loss_alpha
    beta = args.loss_beta
    loss_options = {
        'SCE':
        SCELoss(alpha=alpha, beta=beta, num_classes=num_classes),
        'CE':
        torch.nn.CrossEntropyLoss(),
        'MAE':
        MeanAbsoluteError(scale=alpha, num_classes=num_classes),
        'GCE':
        GeneralizedCrossEntropy(num_classes=num_classes, q=args.q),
        'NCE+RCE':
        NCEandRCE(alpha=alpha, beta=beta, num_classes=num_classes),
        'JSDissect':
        JSDissect(num_classes, args.js_weights, args.dissect_js),
        'LS':
        LabelSmoothing(num_classes=num_classes, t=alpha),
        'JSWC':
        JensenShannonDivergenceWeightedCustom(num_classes=num_classes,
                                              weights=args.js_weights),
        'JSWCS':
        JensenShannonDivergenceWeightedScaled(num_classes=num_classes,
                                              weights=args.js_weights),
        'JSNoConsistency':
        JensenShannonNoConsistency(num_classes=num_classes,
                                   weights=args.js_weights),
        'bootstrap':
        Bootstrapping(num_classes=num_classes, t=alpha)
    }

    if args.loss in loss_options:
        criterion = loss_options[args.loss]
    else:
        raise ("Unknown loss")
    return criterion


# ----
# Based on https://github.com/pytorch/pytorch/blob/0c474d95d9cdd000035dc9e3cd241ba928a08230/aten/src/ATen/native/Loss.cpp#L79
def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()


def custom_ce(prediction, target):
    output_pos = -target * prediction
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()


class JSDissect(torch.nn.Module):

    def __init__(self, num_classes, weights, type):
        super(JSDissect, self).__init__()

        self.num_classes = num_classes
        self.type = type
        self.scale = -1.0 / (0.5 * np.log(0.5))
        if type is not None:
            self.doScale = type.__contains__('s')

    def _get_dissection_loss(self, predictions, labels):

        preds = F.softmax(predictions, dim=1)
        labels = F.one_hot(labels, self.num_classes).float()
        labels_log = labels.clamp(1e-7, 1.0).log()

        distribs = [labels, preds]
        mean_distrib = sum(distribs) / len(distribs)
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        pred_log = preds.clamp(1e-7, 1.0).log()

        if self.type.__contains__('a'):  # I: KL
            train_loss = custom_kl_div(pred_log, labels)
        elif self.type.__contains__('b'):  # J: SymKL KL(p,q) + KL(q,p)
            train_loss = 0.5 * (custom_kl_div(pred_log, labels) +
                                custom_kl_div(labels_log, preds))
        elif self.type.__contains__('c'):  # K: Smoothed KL KL(p, (p+q)/2)
            train_loss = custom_kl_div(mean_distrib_log, labels)
        elif self.type.__contains__(
                'd'):  # L: KL(p, (p+q)/2) + KL(q, (p+q)/2) = 2*JS
            train_loss = 0.5 * (custom_kl_div(mean_distrib_log, labels) +
                                custom_kl_div(mean_distrib_log, preds))
        elif self.type.__contains__(
                'e'
        ):  # 1/sqrt(0.5) * sqrt(JS): sqrt(KL(p, (p+q)/2) + KL(q, (p+q)/2))
            train_loss = torch.sqrt(0.5 *
                                    (custom_kl_div(mean_distrib_log, labels) +
                                     custom_kl_div(mean_distrib_log, preds)))
        elif self.type.__contains__(
                'f'):  # K': reversed smoothed KL: KL(q, (p+q)/2)
            train_loss = custom_kl_div(mean_distrib_log, preds)
        elif self.type.__contains__(
                'g'
        ):  # K'': reversed smoothed CE: CE(q, (p+q)/2) (no entropy reg)
            train_loss = custom_ce(mean_distrib_log, preds)
        elif self.type.__contains__(
                'h'):  # K''': smoothed CE -H(p): KL(p, (p+q)/2) - H(q)
            train_loss = custom_kl_div(mean_distrib_log, labels) - custom_ce(
                pred_log, preds)
        elif self.type.__contains__('i'):  # KL': KL(q, p)
            train_loss = custom_kl_div(labels_log, preds)
        else:
            assert False

        if self.doScale:
            train_loss *= self.scale

        return train_loss

    def forward(self, pred, labels):
        return self._get_dissection_loss(pred, labels)


class JensenShannonDivergenceWeightedCustom(torch.nn.Module):

    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedCustom, self).__init__()
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(' ')]
        assert abs(1.0 - sum(self.weights)) < 0.001

    def forward(self, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([
            w * custom_kl_div(mean_distrib_log, d)
            for w, d in zip(self.weights, distribs)
        ])
        return jsw


class JensenShannonDivergenceWeightedScaled(torch.nn.Module):

    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(' ')]

        self.scale = -1.0 / ((1.0 - self.weights[0]) * np.log(
            (1.0 - self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001

    def forward(self, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))
        # import ipdb; ipdb.set_trace()
        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([
            w * custom_kl_div(mean_distrib_log, d)
            for w, d in zip(self.weights, distribs)
        ])
        return self.scale * jsw


class JensenShannonNoConsistency(torch.nn.Module):

    def __init__(self, num_classes, weights):
        super(JensenShannonNoConsistency, self).__init__()
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(' ')]

        self.scale = -1.0 / ((1.0 - self.weights[0]) * np.log(
            (1.0 - self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001

    def forward(self, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))

        # Take average of predictions
        preds_mean = sum([w * d for w, d in zip(self.weights[1:], preds)])
        weights = [self.weights[0], 1.0 - self.weights[0]]

        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels, preds_mean]

        mean_distrib = sum([w * d for w, d in zip(weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([
            w * custom_kl_div(mean_distrib_log, d)
            for w, d in zip(weights, distribs)
        ])
        return self.scale * jsw


class LabelSmoothing(torch.nn.Module):

    def __init__(self, num_classes, t):
        super(LabelSmoothing, self).__init__()
        self.num_classes = num_classes
        self.t = t

    def forward(self, pred, labels):
        # Create smoothed labels
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        uniform = torch.ones_like(labels_onehot) / float(self.num_classes)
        labels_smooth = (1.0 - self.t) * labels_onehot + self.t * uniform

        pred_log = F.log_softmax(pred, dim=1)
        return F.kl_div(pred_log, labels_smooth, reduction='batchmean')


class Bootstrapping(torch.nn.Module):

    def __init__(self, num_classes, t):
        super(Bootstrapping, self).__init__()
        self.num_classes = num_classes
        self.t = t

    def forward(self, pred, labels):
        # Create smoothed labels
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        prediction = F.softmax(pred, dim=1)
        labels_smooth = (1.0 - self.t) * labels_onehot + self.t * prediction

        pred_log = prediction.clamp(1e-7, 1.0).log()

        return custom_ce(pred_log, labels_smooth)


# ---------------------------------------------------------------------------------
# Implementation of baseline losses from
# https://github.com/HanxunHuangLemonBear/Active-Passive-Losses/blob/master/loss.py
# ---------------------------------------------------------------------------------


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class GeneralizedCrossEntropy(torch.nn.Module):

    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q
        print("q:", q, ", type:", type(q))

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1),
                              self.q)) / self.q
        return gce.mean()


class MeanAbsoluteError(torch.nn.Module):

    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * mae.mean()


class NormalizedCrossEntropy(torch.nn.Module):

    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * nce.mean()


class ReverseCrossEntropy(torch.nn.Module):

    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NCEandRCE(torch.nn.Module):

    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)
