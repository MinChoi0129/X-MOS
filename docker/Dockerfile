# Environment for X-MOS on recent GPUs (e.g. RTX PRO 6000 Blackwell needs CUDA >= 12.8).
# Build from the root of the repository: docker build -f docker/Dockerfile -t xmos .
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 git ninja-build build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# MinkowskiEngine is not maintained anymore, it needs a few patches for CUDA 12 and PyTorch 2.
# No GPU is visible while building, so the target architectures have to be given explicitly.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ARG MAX_JOBS=16
COPY docker/patch_minkowski_engine.sh /tmp/
RUN git clone --depth 1 https://github.com/NVIDIA/MinkowskiEngine /opt/MinkowskiEngine \
    && cd /opt/MinkowskiEngine && bash /tmp/patch_minkowski_engine.sh \
    && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} python setup.py install --blas=openblas --force_cuda \
    && rm -rf build

# The code is not installed, mount the repository here and run the scripts from its root
WORKDIR /workspace/X-MOS
