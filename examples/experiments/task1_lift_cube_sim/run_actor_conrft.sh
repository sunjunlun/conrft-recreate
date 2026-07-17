#!/usr/bin/env bash
# Phase 2 - Actor 进程: 在仿真环境中执行策略, 采集数据并发送给 Learner.
ulimit -n 65536
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=task1_lift_cube_sim \
    --checkpoint_path=/home/robot/sjl/conrft-main/examples/experiments/task1_lift_cube_sim/conrft \
    --actor
