#!/usr/bin/env bash
# Phase 2 - Learner 进程: 从 buffer 采样训练, 把更新后的参数推给 Actor.
ulimit -n 65536
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \

python ../../train_conrft_octo.py "$@" \
    --exp_name=task1_lift_cube_sim \
    --checkpoint_path=/home/robot/sjl/conrft-main/examples/experiments/task1_lift_cube_sim/conrft \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=../../demo_data/task1_lift_cube_sim_30_demos.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner
