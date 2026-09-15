#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

task_name=${1:-adroit_pen}
addition_info=${2:-paper-full}
seed=${3:-0}
gpu_ids=${4:-0}
python_bin=${PYTHON_BIN:-python3}
torchrun_bin=${TORCHRUN_BIN:-torchrun}

exp_name="${task_name}-rgp-dynaslots-${addition_info}"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="data/${task_name}_expert.zarr"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu_ids}"

IFS=',' read -ra gpu_list <<< "${gpu_ids}"
num_gpus=${#gpu_list[@]}
launcher=("${python_bin}")
if (( num_gpus > 1 )); then
    launcher=("${torchrun_bin}" --standalone --nproc_per_node="${num_gpus}")
fi

"${launcher[@]}" pretrain.py --config-name=rgp_dynaslots.yaml \
    task="${task_name}" \
    hydra.run.dir="${run_dir}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    training.resume="${RESUME:-false}" \
    exp_name="${exp_name}" \
    logging.mode=offline \
    checkpoint.save_ckpt=true \
    task.dataset.zarr_path="${zarr_path}"
