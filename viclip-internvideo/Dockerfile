# Базовый образ с PyTorch и CUDA 12.0
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 🛠 Установка зависимостей
RUN apt update && apt install -y ffmpeg git wget build-essential

RUN pip install --upgrade pip
RUN pip install \
    jupyter \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    opencv-python \
    transformers \
    sentencepiece \
    torchcodec \
    torchvision \
    tqdm \
    pysubs2

# Клонируем репозиторий 
WORKDIR /workspace
RUN git clone https://github.com/OpenGVLab/InternVideo.git

# ViCLIP / InternVideo (если есть setup.py)
WORKDIR /workspace/InternVideo
