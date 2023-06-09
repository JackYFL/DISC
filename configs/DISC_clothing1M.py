algorithm = 'DISC'
# dataset param
dataset = 'clothing1M'
input_channel = 3
num_classes = 14
root = '/data/yfli/Clothing1M'
noise_type = 'sym'
percent = 0.8
seed = 123
loss_type = 'ce'
# model param
model1_type = 'resnet50'
model2_type = 'none'
# train param
gpu = '1'
batch_size = 32
lr = 0.01
# lr = 0.0004
epochs = 100
num_workers = 4
epoch_decay_start = 30
alpha = 5.0
# result param
save_result = True