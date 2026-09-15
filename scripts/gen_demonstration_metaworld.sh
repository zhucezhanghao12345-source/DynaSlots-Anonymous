#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

task_name=${1:-bin-picking}
gpu_id=${2:-0}

CUDA_VISIBLE_DEVICES="${gpu_id}" ${PYTHON_BIN:-python3} third_party/Metaworld/gen_demonstration_expert.py \
    --env_name "${task_name}" \
    --num_episodes 25 \
    --root_dir data
