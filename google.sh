#!/bin/bash

# 创建存储输出文件的目录
mkdir -p res

# 在后台运行训练命令，并将输出重定向到不同的文件
python3 train_cifar.py > res/pgd.txt 2>&1 &
python3 train_cifar.py --attack fgsm > res/fgsm.txt 2>&1 &
python3 train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1 > res/pgd_cutout.txt 2>&1 &
python3 train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > res/fgsm_cutout.txt 2>&1 &
python3 train_cifar.py --model WideResNet > res/pgd_WideResNet.txt 2>&1 &
python3 train_cifar.py --model WideResNet --attack fgsm > res/fgsm_WideResNet.txt 2>&1 &
python3 train_cifar.py --model WideResNet --cutout --cutout-len 16 --mixup --mixup-alpha 1 > res/pgd_cutout_WideResNet.txt 2>&1 &
python3 train_cifar.py --model WideResNet --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > res/fgsm_cutout_WideResNet.txt 2>&1 &
python3 train_cifar.py --lr-schedule linear > res/pgd_linear.txt 2>&1 &
python3 train_cifar.py --lr-schedule superconverge > res/pgd_superconverge.txt 2>&1 &
python3 train_cifar.py --lr-schedule piecewisesmoothed > res/pgd_piecewisesmoothed.txt 2>&1 &
python3 train_cifar.py --lr-schedule piecewisezoom > res/pgd_piecewisezoom.txt 2>&1 &
python3 train_cifar.py --lr-schedule onedrop > res/pgd_onedrop.txt 2>&1 &
python3 train_cifar.py --lr-schedule multipledecay > res/pgd_multipledecay.txt 2>&1 &
python3 train_cifar.py --lr-schedule cosine > res/pgd_cosine.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule linear > res/fgsm_linear.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule superconverge > res/fgsm_superconverge.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule piecewisesmoothed > res/fgsm_piecewisesmoothed.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule piecewisezoom > res/fgsm_piecewisezoom.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule onedrop > res/fgsm_onedrop.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule multipledecay > res/fgsm_multipledecay.txt 2>&1 &
python3 train_cifar.py --attack fgsm --lr-schedule cosine > res/fgsm_cosine.txt 2>&1 &
echo "All training processes are started in background."
