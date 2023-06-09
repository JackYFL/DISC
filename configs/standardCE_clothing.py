algorithm = 'StandardCE'
# dataset param
dataset = 'clothing1M'
input_channel = 3
num_classes = 10
root = '/data/yfli/Clothing1M'
noise_type = 'sym'
percent = 0.2
seed = 1
loss_type = 'ce'
# model param
model1_type = 'resnet50'
model2_type = 'none'
# train param
gpu = '1'
batch_size = 32
lr = 0.0001
epochs = 80
num_workers = 4
adjust_lr = 1
epoch_decay_start = 40
# result param
save_result = True
