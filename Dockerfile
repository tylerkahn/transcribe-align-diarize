FROM ghcr.io/astral-sh/uv:python3.10-bookworm
RUN apt-get update && apt-get install -y ffmpeg libsox-dev python3-dev build-essential gcc g++ clang curl

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# Update package list and install CUDA 12.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-12-8 \
    cudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH="/usr/local/cuda-12.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}"


RUN uv pip install --system \
    'huggingface_hub[hf-transfer]' \
    'numpy (>=1.24.3,<2.0.0)' \
    'faster-whisper' \
    'beam-client>=0.2.153' \
    'torchaudio>=2.6.0' \
    'torch>=2.7.0' \
    'fastapi[standard]>=0.115.12' \
    'git+https://github.com/tylerkahn/whisper-diarization' \
    'requests'

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV __TOUCH=3
