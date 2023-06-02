algorithm = 'Mixup'
# dataset param

dataset = 'animal10N'
adjust_lr = 1
if dataset == 'cifar-10':
    input_channel = 3
    num_classes = 10
    root = '/data/yfli/CIFAR10'
    noise_type = 'sym'
    percent = 0.2
    seed = 1
    loss_type = 'ce'
    # model param
    model1_type = 'resnet18'
    model2_type = 'none'
    # train param
    gpu = '1'
    batch_size = 128
    lr = 0.001
    epochs = 200
    num_workers = 4
    epoch_decay_start = 80
    # result param
    save_result = True

elif dataset=='animal10N':
    dataset = 'animal10N'
    input_channel = 3
    num_classes = 14
    root = '/data/yfli/Animal10N'
    noise_type = 'sym'
    percent = 0.8
    seed = 1
    loss_type = 'ce'
    # model param
    model1_type = 'VGG-19'
    model2_type = 'none'
    # train param
    gpu = '1'
    batch_size = 64
    lr = 0.01
    # lr = 0.0004[]
    epochs = 120
    num_workers = 4
    epoch_decay_start = 30
    alpha = 5.0
    # result param
    save_result = True

elif dataset == 'food101N':
    # dataset param
    dataset = 'food101n'
    input_channel = 3
    num_classes = 14
    root = '/data/yfli/Food101N/data/food-101/'
    noise_type = 'sym'
    percent = 0.8
    seed = 1
    loss_type = 'ce'
    # model param
    model1_type = 'resnet50'
    model2_type = 'none'
    # train param
    gpu = '1'
    batch_size = 32
    lr = 0.01
    epochs = 100
    num_workers = 4
    epoch_decay_start = 30
    alpha = 5.0
    # result param
    save_result = True