model_name='DISC'
noise_type1='sym' 
noise_type2='asym' 
gpuid='2'
seed='1'
save_path='./logs/'
data_path='/data/yfli/tiny-imagenet-200'
config_path='./configs/DISC_CIFAR.py'
dataset='tiny_imagenet'
num_classes=200
noise_rates=(0 0.2 0.5)

for noise_rate in "${noise_rates[@]}"
do
    python main.py -c=$config_path --save_path=$save_path --noise_type=$noise_type1 --seed=$seed --gpu=$gpuid --percent=$noise_rate --dataset=$dataset --num_classes=$num_classes  --root=$data_path --model_name=disc
done

noise_rate=0.45
python main.py -c=$config_path --save_path=$save_path --noise_type=$noise_type2 --seed=$seed --gpu=$gpuid --percent=$noise_rate --dataset=$dataset --num_classes=$num_classes  --root=$data_path --model_name=disc
