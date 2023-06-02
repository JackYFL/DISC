model_name='DISC'
gpuid='0'
seed=123
save_path='./logs/'
data_path='/data/yfli/Clothing1M/'
config_path='./configs/DISC_clothing1M.py'
dataset='clothing1M'
num_classes=14

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes
            