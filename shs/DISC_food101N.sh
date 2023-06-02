model_name='DISC'
gpuid='3'
seed=123
save_path='./logs/'
data_path='/data/yfli/Food101N/data/food-101/'
config_path='./configs/DISC_food101N.py'
dataset='food101N'
num_classes=101

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes
            
