algorithm = 'MetaLearning'
# dataset param
dataset = 'cifar-10'
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
lr = 0.2
meta_lr = 0.02
epochs = 200
num_workers = 4
adjust_lr = 1
epoch_decay_start = 80
start_epoch = 1
alpha = 0.2
start_iter = 0
mid_iter = 391 * 20
num_fast = 10
perturb_ratio = 0.5
eps = 0.99
# result param
save_result = True