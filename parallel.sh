#!/bin/bash

#SBATCH --job-name=parallel
#SBATCH --partition=gpu  # 或者您可以选择其他可用的 GPU 分区
#SBATCH --gres=gpu:4  # 请求GPU个数
#SBATCH --output=parallel.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<yuelin.xu@cispa.de>
#SBATCH --time=166:00:00
#SBATCH --container-image pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime


# 在后台运行训练命令，并将输出重定向到不同的文件
python ~/CISPA-home/RML/parallel.py
