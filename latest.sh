#!/bin/bash

#SBATCH --job-name=latest
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:6  # 请求GPU个数
#SBATCH --output=latest.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yuelin.xu@cispa.de>
#SBATCH --time=166:00:00
#SBATCH --container-image pytorch/pytorch:latest

python ~/CISPA-home/RML/train_cifar.py --lr-schedule piecewisezoom > ~/CISPA-home/RML/latest_res/pgd_piecewisezoom.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --lr-schedule piecewisesmoothed > ~/CISPA-home/RML/latest_res/pgd_piecewisesmoothed.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --lr-schedule multipledecay > ~/CISPA-home/RML/latest_res/pgd_multipledecay.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --lr-schedule cosine > ~/CISPA-home/RML/latest_res/pgd_cosine.txt 2>&1

python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule linear > ~/CISPA-home/RML/latest_res/fgsm_linear.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule superconverge > ~/CISPA-home/RML/latest_res/fgsm_superconverge.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule piecewisesmoothed > ~/CISPA-home/RML/latest_res/fgsm_piecewisesmoothed.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule piecewisezoom > ~/CISPA-home/RML/latest_res/fgsm_piecewisezoom.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule onedrop > ~/CISPA-home/RML/latest_res/fgsm_onedrop.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule multipledecay > ~/CISPA-home/RML/latest_res/fgsm_multipledecay.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule cosine > ~/CISPA-home/RML/latest_res/fgsm_cosine.txt 2>&1

python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/latest_res/fgsm_cutout_WideResNet.txt 2>&1

python ~/CISPA-home/RML/train_cifar.py --lr-schedule linear > ~/CISPA-home/RML/latest_res/pgd_linear.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --lr-schedule superconverge > ~/CISPA-home/RML/latest_res/pgd_superconverge.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --lr-schedule onedrop > ~/CISPA-home/RML/latest_res/pgd_onedrop.txt 2>&1

python ~/CISPA-home/RML/train_cifar.py
python ~/CISPA-home/RML/train_cifar.py --attack fgsm > ~/CISPA-home/RML/latest_res/fgsm.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/latest_res/pgd_cutout.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/latest_res/fgsm_cutout.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --model WideResNet > ~/CISPA-home/RML/latest_res/pgd_WideResNet.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm > ~/CISPA-home/RML/latest_res/fgsm_WideResNet.txt 2>&1
python ~/CISPA-home/RML/train_cifar.py --model WideResNet --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/latest_res/pgd_cutout_WideResNet.txt 2>&1

