#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

task_name=${1:-adroit_pen}
run_name=${2:-run1}
seed=${3:-0}
gpu_id=${4:-0}

pretrain_exp="${task_name}-dynaslots-${run_name}"
policy_exp="${task_name}-dynaslots_policy-${run_name}"
pretrain_checkpoint="data/outputs/${pretrain_exp}_seed${seed}/checkpoints/pretrain_latest.ckpt"
run_dir="data/outputs_policy/${policy_exp}_seed${seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${gpu_id}"

${PYTHON_BIN:-python3} eval.py --config-name=dynaslots_policy.yaml \
    task="${task_name}" \
    task_whole_name="${task_name}" \
    hydra.run.dir="${run_dir}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    exp_name="${policy_exp}" \
    logging.mode=offline \
    policy.checkpoint="${pretrain_checkpoint}"
