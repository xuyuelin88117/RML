#!/bin/bash

#SBATCH --job-name=RML-sh
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:4  # 请求GPU个数
#SBATCH --time=16:00:00  # 或者您需要的时间
#SBATCH --output=job-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<yuelin.xu@cispa.de>

# 运行您的 Python 脚本
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/RML/train_cifar.py
python ~/CISPA-home/RML/train_cifar.py --attack fgsm
python ~/CISPA-home/RML/train_cifar.py --cutout True --cutout-len 16 --mixup True --mixup-alpha 1
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --cutout True --cutout-len 16 --mixup True --mixup-alpha 1