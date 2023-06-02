import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from .config import Config
import random
import torch.backends.cudnn as cudnn
import json
import pickle


def get_gt_labels(dataset, root):
    if dataset == 'cifar-10':
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]
        base_folder = 'cifar-10-batches-py'
    elif dataset == 'cifar-100':
        train_list = [
            ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]
        base_folder = 'cifar-100-python'
    targets = []
    for file_name, _ in train_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])
    return targets


def load_config(filename: str = None, _print: bool = True):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict

    if _print == True:
        print_config(config)

    return config


def print_config(config):
    print('---------- params info: ----------')
    for k, v in config.items():
        print(k, ' : ', v)
    print('---------------------------------')


def get_log_name(config, path='./log2'):
    log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
                str(config['percent']) + '_seed' + str(config['seed']) + '.json'

    if not osp.exists(path):
        os.makedirs(path)
    data_root = path + '/' + config['dataset'] + '/' + config['noise_type']
    if not osp.exists(data_root):
        os.makedirs(data_root)
    log_name = osp.join(data_root, log_name)

    return log_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def save_results(config, last_ten, best_acc, best_epoch, jsonfile):
    result_dict = config
    result_dict['last10_acc_mean'] = last_ten.mean()
    result_dict['last10_acc_std'] = last_ten.std()
    result_dict['best_acc'] = best_acc
    result_dict['best_epoch'] = best_epoch
    with open(jsonfile, 'w') as out:
        json.dump(result_dict, out, sort_keys=False, indent=4)


def plot_results(epochs, test_acc, plotfile):
    plt.style.use('ggplot')
    plt.plot(np.arange(1, epochs), test_acc, label='scratch - acc')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 20)))  # train epochs
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 101, 10))  # Acc range: [0, 100]
    plt.ylabel('Acc divergence')
    plt.savefig(plotfile)


def get_test_acc(acc):
    return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc