#!/bin/bash

#SBATCH --job-name=RML-model
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:2  # 请求GPU个数
#SBATCH --output=job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yuelin.xu@cispa.de>

#SBATCH --container-image pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# 运行您的 Python 脚本
python ~/CISPA-home/RML/train_cifar.py
python ~/CISPA-home/RML/train_cifar.py --attack fgsm
python ~/CISPA-home/RML/train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1
python ~/CISPA-home/RML/train_cifar.py --model WideResNet
python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm
python ~/CISPA-home/RML/train_cifar.py --model WideResNet --cutout --cutout-len 16 --mixup --mixup-alpha 1
python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1
