#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${REPO_DIR}/.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-cortex-fusion:cu128}
GPU_SPEC=${GPU_SPEC:-all}
SHM_SIZE=${SHM_SIZE:-16g}
PROJECT_MOUNT=${PROJECT_MOUNT:-${PROJECT_ROOT}}
CONTAINER_PROJECT_ROOT=${CONTAINER_PROJECT_ROOT:-${PROJECT_ROOT}}
CONTAINER_REPO_DIR=${CONTAINER_REPO_DIR:-${CONTAINER_PROJECT_ROOT}/cortex_fusion}
HOME_DIR=${HOME_DIR:-/tmp}

if [ "$#" -gt 0 ]; then
  CMD=("$@")
else
  CMD=(bash)
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
fi

docker run --rm \
  "${TTY_ARGS[@]}" \
  --gpus "${GPU_SPEC}" \
  --ipc=host \
  --shm-size "${SHM_SIZE}" \
  --user "$(id -u):$(id -g)" \
  -e HOME="${HOME_DIR}" \
  -e MPLBACKEND=Agg \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="${CONTAINER_REPO_DIR}" \
  -v "${PROJECT_MOUNT}:${CONTAINER_PROJECT_ROOT}" \
  -w "${CONTAINER_REPO_DIR}" \
  "${IMAGE_NAME}" \
  "${CMD[@]}"
