#!/bin/bash

# 在后台运行训练命令，并将输出重定向到不同的文件
python train_cifar.py > pgd.txt 2>&1 &
python train_cifar.py --attack fgsm > fgsm.txt 2>&1 &
python train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1 > pgd_cutout.txt 2>&1 &
python train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > fgsm_cutout.txt 2>&1 &
python train_cifar.py --model WideResNet > pgd_WideResNet.txt 2>&1 &
python train_cifar.py --model WideResNet --attack fgsm > fgsm_WideResNet.txt 2>&1 &
python train_cifar.py --model WideResNet --cutout --cutout-len 16 --mixup --mixup-alpha 1 > pgd_cutout_WideResNet.txt 2>&1 &
python train_cifar.py --model WideResNet --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > fgsm_cutout_WideResNet.txt 2>&1 &
python train_cifar.py --lr-schedule linear > pgd_linear.txt 2>&1 &
python train_cifar.py --lr-schedule superconverge > pgd_superconverge.txt 2>&1 &
python train_cifar.py --lr-schedule piecewisesmoothed > pgd_piecewisesmoothed.txt 2>&1 &
python train_cifar.py --lr-schedule piecewisezoom > pgd_piecewisezoom.txt 2>&1 &
python train_cifar.py --lr-schedule onedrop > pgd_onedrop.txt 2>&1 &
python train_cifar.py --lr-schedule multipledecay > pgd_multipledecay.txt 2>&1 &
python train_cifar.py --lr-schedule cosine > pgd_cosine.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule linear > fgsm_linear.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule superconverge > fgsm_superconverge.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule piecewisesmoothed > fgsm_piecewisesmoothed.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule piecewisezoom > fgsm_piecewisezoom.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule onedrop > fgsm_onedrop.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule multipledecay > fgsm_multipledecay.txt 2>&1 &
python train_cifar.py --attack fgsm --lr-schedule cosine > fgsm_cosine.txt 2>&1 &
echo "All training processes are started in background."
