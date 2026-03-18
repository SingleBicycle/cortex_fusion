ARG BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG CUDA_FLAVOR=cu128

ENV PATH=/opt/venv/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    tini \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/cortex_fusion

COPY requirements.docker.txt /tmp/requirements.docker.txt

RUN python -m pip install --index-url https://download.pytorch.org/whl/${CUDA_FLAVOR} \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
 && python -m pip install -r /tmp/requirements.docker.txt

COPY . /workspace/cortex_fusion

RUN python -m pip install -e /workspace/cortex_fusion/third_party/UDIP-ViT

ENV PYTHONPATH=/workspace/cortex_fusion

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
