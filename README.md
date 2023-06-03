# DISC: Learning from Noisy Labels via Dynamic Instance-Specific Selection and Correction

[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_DISC_Learning_From_Noisy_Labels_via_Dynamic_Instance-Specific_Selection_and_CVPR_2023_paper.pdf)], [[sup](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Li_DISC_Learning_From_CVPR_2023_supplemental.pdf)], [[video](https://www.youtube.com/watch?v=YGWGU3m4whc)], [[poster](./assets/DISC_poster.png)]

This repository is the official implementation of the **CVPR2023** paper "DISC: Learning from Noisy Labels via Dynamic Instance-Specific Selection and Correction". This repository includes several baseline methods and supports almost all commonly used benchmarks and label noise types. It can serve as a library for LNL models.

<b>Title</b>: DISC: Learning from Noisy Labels via Dynamic Instance Specific Selection and Correction \
<b>Authors</b>: Yifan Li, Hu Han, Shiguang Shan, Xilin Chen \
<b>Institute</b>: Institute of Computing Technology, Chinese Academy of Sciences 

## Citing DISC

If you find this repo is useful, please cite the following BibTeX entry. Thank you very much!

```
@InProceedings{Li_2023_DISC,
    author    = {Li, Yifan and Han, Hu and Shan, Shiguang and Chen, Xilin},
    title     = {DISC: Learning From Noisy Labels via Dynamic Instance-Specific Selection and Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24070-24079}
}
```


## Content
- [DISC](#disc-learning-from-noisy-labels-via-dynamic-instance-specific-selection-and-correction)
  - [1. Abstract](#1-abstract)
  - [2. Requirements](#2-requirements)
  - [3. Datasets](#3-datasets)
  - [4. Reproduce the results of DISC](#4-reproduce-the-results-of-disc)
  - [5. Other methods](#5-other-methods)
  - [6. Reference](#6-reference)
  - [7. Contact](#7-contact)
  
<p align="center">
  <img width="90%" src="./assets/DISC.png" alt="The framework of DISC"/>
</p>

## 1. Abstract

Existing studies indicate that deep neural networks (DNNs) can eventually memorize the label noise.  We observe that the memorization strength of DNNs towards each instance is different and can be represented by the confidence value, which becomes larger and larger during the training process. Based on this, we propose a Dynamic Instance-specific Selection and Correction method (DISC) for learning from noisy labels (LNL).  We first use a two-view-based backbone for image classification, obtaining confidences for each image from two views. Then we propose a dynamic threshold strategy for each instance, based on the momentum of each instance's memorization strength in previous epochs to select and correct noisy labeled data. Benefiting from the dynamic threshold strategy and two-view learning, we can effectively group each instance into one of the three subsets (i.e., clean, hard, and purified) based on the prediction consistency and discrepancy by two views at each epoch. Finally, we employ different regularization strategies to conquer subsets with different degrees of label noise, improving the whole network's robustness. Comprehensive evaluations on three controllable and four real-world LNL benchmarks show that our method outperforms the state-of-the-art (SOTA) methods to  leverage useful information in noisy data while alleviating the pollution of label noise.


## 2. Requirements
The code requires `python>=3.7` and the following packages.
```
torch==1.8.0
torchvision==0.9.0
numpy==1.19.4
scipy==1.6.0
addict==2.4.0
tqdm==4.64.0
nni==2.5
```
These packages can be installed directly by running the following command:
```
pip install -r requirements.txt
```
Note that all the experiments are conducted under one single <b>RTX 3090</b>, so the results may be a little different with the original paper when you use a different gpu.

## 3. Datasets
This code includes seven datasets including:
CIFAR-10, CIFAR-100, Tiny-ImageNet, Animals-10N, Food-101, Mini-WebVision (top-50 classes from WebVisionV1.0 (training set and validation set) and ILSVRC-2012 (only validation set)) and Clothing1M. 

|Datasets|Download links|
| --------- | ---- |
|CIFAR-10|[link](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)|
|CIFAR-100|[link](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)|
|Tiny-ImageNet|[link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)|
|Animals-10N|[link](https://forms.gle/8mbmbNgDFQ2rA1fLA)|
|Food-101|[link](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)|
|WebVision V1.0|[link](https://data.vision.ee.ethz.ch/cvl/webvision/download.html)|
|ILSVRC-2012|[link](https://image-net.org/challenges/LSVRC/2012/index.php)|
|Clothing1M|[link](https://github.com/Cysu/noisy_label)|

If you want to run one of the datasets, please <span style="color:#0099be">download it into your data directory and change the dataset path in bash scripts (see the following section)</span>.

## 4. Reproduce the results of DISC
In order to reproduce the results of DISC, you need to change the hyper-parameters in bash scripts directory ([shs/](./shs/)) for different datasets.

### 4.1 Synthetic dataset (CIFAR-10, CIFAR-100, Tiny-ImageNet)

For instance, for the synthetic label noise dataset CIFAR10, the example of script is shown as follows:
```bash
model_name='DISC' # the extra model name of the algorithm
noise_type='ins' # the label noise type, which could be: 'ins', 'sym', 'asym'
gpuid='1' # the gpu to assign
seed='1' # the random seed
save_path='./logs/' # the directory for saving logs
data_path='/data/yfli/CIFAR10' # the directory of dataset
config_path='./configs/DISC_CIFAR.py' # the configuration file of algorithm for different datasets
dataset='cifar-10' # the name of dataset
num_classes=10 # the class number of dataset
noise_rates=(0.2 0.4 0.6) # the noise rate for synthetic label noise datasets

for noise_rate in "${noise_rates[@]}"
do
    python main.py -c=$config_path  --save_path=$save_path --noise_type=$noise_type --seed=$seed --gpu=$gpuid --percent=$noise_rate --dataset=$dataset --num_classes=$num_classes  --root=$data_path --model_name=disc
done
```
If you want to run the results of DISC on CIFAR-10 with inst. noise, please <span style="color:#0099be">change the data directory as yours</span>  and run the following command:
```shell
bash shs/DISC_cifar10.sh
```
Furthermore, there are <span style="color:#0099be">three types of label noise</span> to choose from: symmetric noise ('<span style="color:#0099be">sym</span>'), asymmetric noise ('<span style="color:#0099be">asym</span>'), and instance noise ('<span style="color:#0099be">ins</span>'). You can also <span style="color:#0099be">adjust the noise ratio</span> by changing the 'noise_rates' hyper-parameter.

### 4.2 Real-world noise datasets (Animals-10N, Food-101, Mini-WebVision, Clothing1M)
If you want to run the results of DISC on Animals-10N, the bash scripts can be shown as:
```bash
model_name='DISC' # the extra model name of the algorithm
gpuid='0' # the gpu to assign
save_path='./logs/' # the directory for saving logs
data_path='/data/yfli/Animal10N/' # the directory of the dataset, and you need to change this as yours
config_path='./configs/DISC_animal10N.py'
dataset='animal10N'
num_classes=10

python main.py -c='./configs/DISC_animal10N.py' --save_path=$save_path
               --gpu=$gpuid --model_name=$model_name 
               --root=$data_path 
               --dataset=$dataset --num_classes=$num_classes
```
All you need to do is to change the directory of data path as yours, and run:
```bash
bash shs/DISC_animal10N.sh
```


## 5. Other methods
Currently, there are many other baselines in this database ([algorithms/](./algorithms/)) including Co-learning, Co-teaching, Co-teaching+, Decoupling, ELR, GJS, JoCoR, JointOptim, MetaLearning, Mixup, NL, PENCIL. 

However, these methods are currently only applicable to the CIFAR-10/100 datasets. You can adapt the DISC code to achieve results on other benchmarks according to your needs.

We hope this repository will serve as a codebase for LNL in the future. Anyone who wishes to contribute can do so by submitting a pull request or forking it to their own repository. 

## 6. Reference
This codebase refers to Co-learning [[link](https://github.com/chengtan9907/Co-learning-Learning-from-noisy-labels-with-self-supervision)], DivideMix [[link](https://github.com/LiJunnan1992/DivideMix)], ELR [[link](https://github.com/shengliu66/ELR)], think you all!

## 7. Contact
If you have any other questions, please contact liyifan20g@ict.ac.cn.

## License
This repo is licensed under MIT License.

