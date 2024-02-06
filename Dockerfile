# 基础镜像
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# 安装数据处理和分析库
RUN pip install numpy pandas matplotlib seaborn

# 安装深度学习相关库
RUN pip install torchvision torchaudio torchtext

# 安装机器学习辅助库
RUN pip install scikit-learn

# 安装版本控制工具
RUN apt-get update && apt-get install -y git

# 设置工作目录
WORKDIR /workspace

# 默认执行命令
CMD ["bash"]

