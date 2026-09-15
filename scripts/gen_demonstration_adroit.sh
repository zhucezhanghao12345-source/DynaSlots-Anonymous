#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

task=${1:-pen}
gpu_id=${2:-0}

CUDA_VISIBLE_DEVICES="${gpu_id}" ${PYTHON_BIN:-python3} third_party/VRL3/src/gen_demonstration_expert.py \
    --env_name "${task}" \
    --num_episodes 100 \
    --root_dir data \
    --expert_ckpt_path "third_party/VRL3/ckpts/vrl3_${task}.pt" \
    --img_size 84 \
    --not_use_multi_view \
    --use_point_crop
