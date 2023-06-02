from .noise_datasets import cifar_dataloader
from .clothing1M import clothing_dataloader
from .webvision import webvision_dataloader
from .food101N import food101N_dataloader
from .animal10N import animal10N_dataloader
from .tiny_imagenet import tiny_imagenet_dataloader

__all__ = ('cifar_dataloader', 'clothing_dataloader', 'webvision_dataloader', 
           'food101N_dataloader', 'tiny_imagenet_dataloader', 'animal10N_dataloader')
