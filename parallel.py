import subprocess

# 定义要运行的命令列表
commands = [
    "python ~/CISPA-home/RML/train_cifar.py > ~/CISPA-home/RML/para_res/pgd.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm > ~/CISPA-home/RML/para_res/fgsm.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/para_res/pgd_cutout.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/para_res/fgsm_cutout.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --model WideResNet > ~/CISPA-home/RML/para_res/pgd_WideResNet.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm > ~/CISPA-home/RML/para_res/fgsm_WideResNet.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --model WideResNet --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/para_res/pgd_cutout_WideResNet.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --model WideResNet --attack fgsm --cutout --cutout-len 16 --mixup --mixup-alpha 1 > ~/CISPA-home/RML/para_res/fgsm_cutout_WideResNet.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule linear > ~/CISPA-home/RML/para_res/pgd_linear.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule superconverge > ~/CISPA-home/RML/para_res/pgd_superconverge.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule piecewisesmoothed > ~/CISPA-home/RML/para_res/pgd_piecewisesmoothed.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule piecewisezoom > ~/CISPA-home/RML/para_res/pgd_piecewisezoom.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule onedrop > ~/CISPA-home/RML/para_res/pgd_onedrop.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule multipledecay > ~/CISPA-home/RML/para_res/pgd_multipledecay.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --lr-schedule cosine > ~/CISPA-home/RML/para_res/pgd_cosine.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule linear > ~/CISPA-home/RML/para_res/fgsm_linear.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule superconverge > ~/CISPA-home/RML/para_res/fgsm_superconverge.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule piecewisesmoothed > ~/CISPA-home/RML/para_res/fgsm_piecewisesmoothed.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule piecewisezoom > ~/CISPA-home/RML/para_res/fgsm_piecewisezoom.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule onedrop > ~/CISPA-home/RML/para_res/fgsm_onedrop.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule multipledecay > ~/CISPA-home/RML/para_res/fgsm_multipledecay.txt 2>&1",
    "python ~/CISPA-home/RML/train_cifar.py --attack fgsm --lr-schedule cosine > ~/CISPA-home/RML/para_res/fgsm_cosine.txt 2>&1"
]

# 启动每个命令
processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

# 等待所有命令完成
for proc in processes:
    proc.wait()
