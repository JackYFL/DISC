model_name='DISC'
gpuid='0'
save_path='./logs/'
data_path='/data/yfli/Animal10N/'
config_path='./configs/DISC_animal10N.py'
dataset='animal10N'
num_classes=10

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path \
               --dataset=$dataset --num_classes=$num_classes
            