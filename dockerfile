# 1 · Use a CUDA base image that matches a published wheel
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2 · System packages (Python 3.10 by default on Ubuntu 22.04)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git wget ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# 3 · Python packages – notice the cu123 tag & index URL
# ---- Python packages -------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
    # 1) Point pip to the cu128 sub-index (official wheel set)
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    # 2) Rest of your dependencies
    pip install --no-cache-dir \
        ultralytics opencv-python-headless==4.* numpy pandas tqdm \
        gtts pyttsx3 'carla==0.9.15'

WORKDIR /workspace
CMD ["/bin/bash"]
