# -*- coding: utf-8 -*-
# @Author : Jack (this code is copied from Cheng Tan's codebase "Co-training-based_noisy-label-learning-master". Thank you!)
# @Email  : liyifan20g@ict.ac.cn
# @File   : get model.py


import models

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def get_model(
    model_name: str = 'resnet18',
    input_channel: int = 3,
    num_classes: int = 10,
    device=None,
):

    assert model_name in model_names
    model = models.__dict__[model_name](input_channel=input_channel,
                                        num_classes=num_classes)
    return model.to(device)