#!/bin/bash

#SBATCH --job-name=RML-sh
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:2  # 请求GPU个数
#SBATCH --output=job-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<yuelin.xu@cispa.de>

# 运行您的 Python 脚本
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/RML/train_cifar.py
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/RML/train_cifar.py --attack fgsm
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/RML/train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1
srun --container-image=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
python ~/CISPA-home/RML/train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1
