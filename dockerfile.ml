# ────────────────────────────────────────────────────────────────────────────────
#  CUDA-12.9 + cuDNN runtime base (Ubuntu 22.04)
# ────────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ────────────────────────────────────────────────────────────────────────────────
#  System packages (Python 3.10 default on 22.04)
# ────────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python-is-python3 \
        git wget curl ffmpeg libespeak1 build-essential python3-dev aria2 && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────────────────────
#  Python packages
#    • cu128 wheels match CUDA 12.x drivers
#    • Keep layers small with --no-cache-dir
# ────────────────────────────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip && \
    # --- Core deep-learning stack ---------------------------------------------
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    # --- YOLO ---------------------------------------------------------------
    pip install --no-cache-dir \
        ultralytics==8.3.162 opencv-contrib-python==4.12.0.88 && \
    # --- U-Net + encoders ------------------------------------------------------
    pip install --no-cache-dir \
        segmentation-models-pytorch==0.5.0 albumentations==2.0.8 timm==1.0.16 && \
    # --- Supervision ---------------------------------------------------
    pip install --no-cache-dir \i
        supervision==0.26.0 && \
    # --- Object tracking -------------------------------------------------------
    pip install --no-cache-dir \
        lap cython-bbox filterpy motmetrics \
        git+https://github.com/ifzhang/ByteTrack.git@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 && \
    # --- Text-to-speech & misc -------------------------------------------------
    pip install --no-cache-dir \
        gtts pyttsx3 && \     
    # --- QoL / logging / viz ---------------------------------------------------
    pip install --no-cache-dir \
        tensorboard scikit-image matplotlib shapely \
        tqdm numpy pandas aria2p scipy scikit-learn pillow pyyaml==6.0.2 && \
    # --- Clean pip cache -------------------------------------------------------
    rm -rf /root/.cache/pip

# ────────────────────────────────────────────────────────────────────────────────
#  (Optional) Python runtime flags
# ────────────────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/opt/torch-cache

# ────────────────────────────────────────────────────────────────────────────────
#  Working directory
# ────────────────────────────────────────────────────────────────────────────────
WORKDIR /workspace

# ────────────────────────────────────────────────────────────────────────────────
#  Default entry point
# ────────────────────────────────────────────────────────────────────────────────
CMD ["/bin/bash"]

