#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash scripts/train_dynaslots_policy.sh adroit_pen 0001 0 0
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

task_name=${1:-adroit_pen}
addition_info=${2:-0001}
seed=${3:-0}
gpu_ids=${4:-0}

pretrain_exp="${task_name}-dynaslots-${addition_info}"
exp_name="${task_name}-dynaslots_policy-${addition_info}"
checkpoint="data/outputs/${pretrain_exp}_seed${seed}/checkpoints/pretrain_latest.ckpt"
run_dir="data/outputs_policy/${exp_name}_seed${seed}"
zarr_path="data/${task_name}_expert.zarr"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MUJOCO_HOME:-${HOME}/.mujoco/mujoco210}/bin:/usr/lib/nvidia"

IFS=',' read -ra gpu_list <<< "${gpu_ids}"
num_gpus=${#gpu_list[@]}
python_bin=${PYTHON_BIN:-python3}
torchrun_bin=${TORCHRUN_BIN:-torchrun}
launcher=("${python_bin}")
per_gpu_batch=128
if (( num_gpus > 1 )); then
    launcher=("${torchrun_bin}" --standalone --nproc_per_node="${num_gpus}")
    per_gpu_batch=$((128 / num_gpus))
    if (( per_gpu_batch * num_gpus != 128 )); then
        echo "GPU count ${num_gpus} must divide the global batch size 128" >&2
        exit 2
    fi
fi

"${launcher[@]}" train.py --config-name=dynaslots_policy.yaml \
    task="${task_name}" \
    task_whole_name="${task_name}" \
    hydra.run.dir="${run_dir}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    training.resume=false \
    exp_name="${exp_name}" \
    logging.mode=offline \
    checkpoint.save_ckpt=true \
    task.env_runner.eval_episodes=50 \
    dataloader.batch_size="${per_gpu_batch}" \
    policy.checkpoint="${checkpoint}" \
    task.dataset.zarr_path="${zarr_path}"
