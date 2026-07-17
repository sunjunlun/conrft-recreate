#!/usr/bin/env bash
# Phase 1: 离线预训练 (BC 为主, RL 为辅)
# 只需要 Learner 进程, 不需要 Actor (没有环境交互)
#export TRANSFORMERS_OFFLINE=1
#export HF_HUB_OFFLINE=1
#bash run_learner_conrft_pretrain.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \

python ../../train_conrft_octo.py "$@" \
    --exp_name=task1_lift_cube_sim \
    --checkpoint_path=/home/robot/sjl/conrft-main/examples/experiments/task1_lift_cube_sim/conrft \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=../../demo_data/task1_lift_cube_sim_30_demos.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner
