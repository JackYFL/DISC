# -*- coding: utf-8 -*-
# @Author : Jack (thanks for Karim. Some of the codes refered to his codebase "UNICON-Noisy-Label")
# @Email  : liyifan20g@ict.ac.cn
# @File   : tiny_imagenet.py

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch

from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from .randaugment import TransformFixMatchMedium
from .utils import *

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')


class NoiseDataset(torchvision.datasets.VisionDataset):

    def __init__(
        self,
        noise_type: str = 'none',
        asym_trans: dict = None,
        percent: float = 0.0,
    ) -> None:

        assert percent <= 1.0 and percent >= 0.0
        assert noise_type in ['sym', 'asym', 'ins', 'none']

        self.percent = percent
        self.noise_type = noise_type
        self.asym_trans = asym_trans

        # dataset info
        self.min_target = min(self.targets)
        self.max_target = max(self.targets)
        self.num_classes = len(np.unique(self.targets))
        assert self.num_classes == self.max_target - self.min_target + 1
        self.num_samples = len(self.targets)

        if self.noise_type == 'sym':
            self.symmetric_noise()
        elif self.noise_type == 'asym':
            self.asymmetric_noise()
        elif self.noise_type == 'ins':
            self.instance_noise(tau=self.percent)

    def symmetric_noise(self):
        type = 1

        if type == 1:  # The noisy labels may include clean labels
            indices = np.random.permutation(len(self.data))
            for i, idx in enumerate(indices):
                if i < self.percent * len(self.data):
                    self.targets[idx] = np.random.randint(
                        low=self.min_target,
                        high=self.max_target + 1,
                        dtype=np.int32)
        else:
            random_state = 0
            dataset = 'tiny_imagenet'
            train_noisy_labels, actual_noise_rate = noisify(
                dataset=dataset,
                train_labels=np.array(self.targets).reshape(
                    [len(self.targets), 1]),
                noise_type=self.noise_type,
                noise_rate=self.percent,
                random_state=random_state,
                nb_classes=self.num_classes)
            self.targets = [int(label) for label in train_noisy_labels]
            print("Actual noise rate is %.4f" % actual_noise_rate)

    def asymmetric_noise(self):
        type = 2
        random_state = 0
        dataset = 'tiny_imagenet'
        self.noise_type = 'pairflip'
        train_noisy_labels, actual_noise_rate = noisify(
            dataset=dataset,
            train_labels=np.array(self.targets).reshape([len(self.targets),
                                                         1]),
            noise_type=self.noise_type,
            noise_rate=self.percent,
            random_state=random_state,
            nb_classes=self.num_classes)
        self.targets = [int(label) for label in train_noisy_labels]
        print("Actual noise rate is %.4f" % actual_noise_rate)

    def instance_noise(
        self,
        tau: float = 0.2,
        std: float = 0.1,
        feature_size: int = 3 * 32 * 32,
    ):
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        # np.random.seed(int(seed))
        # torch.manual_seed(int(seed))
        # torch.cuda.manual_seed(int(seed))

        # common-used parameters
        num_samples = self.num_samples
        num_classes = self.num_classes

        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std,
                                            loc=tau,
                                            scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # import ipdb; ipdb.set_trace()
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size,
                            num_classes)).float().to(
                                device)  #K*dim*K, dim=3072
        for i in range(num_samples):
            x, y = self.transform(Image.fromarray(self.data[i])), torch.tensor(
                self.targets[i])
            x = x.to(device)
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)  #classes
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(self.min_target, self.max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.targets)).mean())
        self.targets = new_label


def has_file_allowed_extension(filename: str, extensions: Tuple[str,
                                                                ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory)
                     if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    clsa, class_to_idx = find_classes(directory)
    # print(clsa,class_to_idx)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(
                x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, class_to_idx


class tiny_imagenet_dataloader():

    def __init__(self, root_dir, batch_size, num_workers, noise_type, percent):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_type = noise_type
        self.percent = percent

    def run(self, mode='train'):
        if mode == 'train':
            train_dataset = tiny_imagenet_dataset(root_dir = self.root_dir, noise_mode=self.noise_type, \
                                                  ratio=self.percent, mode=mode)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True)
            return train_loader

        elif mode == 'train_index':
            train_dataset = tiny_imagenet_dataset(root_dir = self.root_dir, noise_mode=self.noise_type, \
                                                  ratio=self.percent, mode=mode)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True)
            return train_loader

        elif mode == 'test':
            test_dataset = tiny_imagenet_dataset(root_dir = self.root_dir, noise_mode=self.noise_type, \
                                                  ratio=self.percent, mode='val')
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)
            return test_loader


class tiny_imagenet_dataset(NoiseDataset):

    def __init__(self, root_dir, ratio, noise_mode, mode='train'):
        self.transform_train_weak = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_train_strong = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # import ipdb; ipdb.set_trace()
        if mode == 'train_index':
            self.transform_train = TransformFixMatchMedium(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.root = root_dir
        self.ratio = ratio
        self.noise_mode = noise_mode
        self.mode = mode
        ### Get the instances and check if it is right
        if 'train' in mode:
            data_folder = '%s/train/' % self.root
            instances, dict_classes = make_dataset(data_folder,
                                                   extensions=IMG_EXTENSIONS)
        elif mode == 'val':
            ## Validation Files
            data_folder = '%s/val/' % self.root
            _, dict_classes = find_classes(data_folder.replace('val', 'train'))
            instances = make_dataset(data_folder, extensions=IMG_EXTENSIONS)
            val_text = '%s/val/val_annotations.txt' % self.root
            val_img_files = '%s/val/images' % self.root
        elif mode == 'test':
            data_folder = '%s/test/' % self.root
            test_instances = make_dataset(data_folder,
                                          extensions=IMG_EXTENSIONS)

        self.data = []
        self.targets = []

        if 'train' in mode:
            for kk in range(len(instances)):
                path_ind = list(instances[kk])[0]
                self.targets.append(int(list(instances[kk])[1]))
                self.data.append(path_ind)
            NoiseDataset.__init__(self,
                                  noise_type=self.noise_mode,
                                  asym_trans=None,
                                  percent=self.ratio)

        elif mode == 'val':
            with open(val_text, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = '%s/' % val_img_files + entry[0]
                    self.targets.append(int(dict_classes[entry[1]]))
                    self.data.append(img_path)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        # print(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.mode == 'train_single':
            img = self.transform_train_weak(image)
            return img, target
        elif self.mode == 'train':
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode == 'train_index' or self.mode == 'train_index_2strong':
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'tripartite':
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'val':
            img = self.transform_test(image)
            return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = tiny_imagenet_dataset(
        root_dir='/old_home/yfli/tiny-imagenet-200',
        ratio=0.2,
        noise_mode='asym',
        mode='train')
