model_name='DISC'
gpuid='0'
seed=10
save_path='./logs/'
data_path='/data/yfli/WebVisionV1.0/'
config_path='./configs/DISC_webvision.py'
dataset='webvision'
num_classes=50

python main.py -c=$config_path --save_path=$save_path\
               --gpu=$gpuid --model_name=$model_name \
               --root=$data_path --seed=$seed\
               --dataset=$dataset --num_classes=$num_classes
            