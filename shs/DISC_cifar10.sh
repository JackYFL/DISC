model_name='DISC'
noise_type='ins' 
gpuid='1'
seed='1'
save_path='./logs/'
data_path='/data/yfli/CIFAR10'
config_path='./configs/DISC_CIFAR.py'
dataset='cifar-10'
num_classes=10
noise_rates=(0.2 0.4 0.6)

for noise_rate in "${noise_rates[@]}"
do
    python main.py -c=$config_path --save_path=$save_path --noise_type=$noise_type --seed=$seed --gpu=$gpuid --percent=$noise_rate --dataset=$dataset --num_classes=$num_classes  --root=$data_path --model_name=disc
done
