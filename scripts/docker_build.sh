#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-cortex-fusion:cu128}
BASE_IMAGE=${BASE_IMAGE:-nvidia/cuda:12.8.1-runtime-ubuntu22.04}
PYTORCH_VERSION=${PYTORCH_VERSION:-2.7.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.1}
CUDA_FLAVOR=${CUDA_FLAVOR:-cu128}

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg PYTORCH_VERSION="${PYTORCH_VERSION}" \
  --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
  --build-arg CUDA_FLAVOR="${CUDA_FLAVOR}" \
  -t "${IMAGE_NAME}" \
  "${REPO_DIR}"
