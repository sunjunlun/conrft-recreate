# task1_lift_cube_sim — ConRFT 仿真复现

本目录在 **不依赖真实 Franka 机械臂** 的前提下,完整复现 ConRFT 训练流程
(数据采集 → 奖励分类器 → Phase-1 离线预训练 → Phase-2 在线训练)。

任务: **抓起立方体并抬起**(简化版的 pick-and-place,作为 pick-banana 的仿真代理)。
仿真后端: [robosuite](https://robosuite.ai/) (基于 MuJoCo)。

---

## 1. 安装依赖

在已经搭好 ConRFT 主环境(参考根目录 `README.md`)之后,额外安装仿真器:

```bash
conda activate conrft
pip install robosuite mujoco
```

可选:加载 Octo 时需要 `octo` 包,并把模型路径写入环境变量
(`collect_sim_demos.py --with_octo` 才会用到):

```bash
pip install octo
export OCTO_PATH=/绝对路径/octo-small
```

> Windows 用户:JAX 在原生 Windows 上几乎无法使用,**建议在 WSL2 / Ubuntu 中运行**。

---

## 2. 文件结构与作用

| 文件 | 作用 |
|---|---|
| `sim_env.py` | `SimFrankaEnv`:接口完全对齐 `FrankaEnv`,内部包装 robosuite 的 `Lift` 任务 |
| `wrapper.py` | `LiftCubeSimEnv`:实现具体任务的成功条件;`GripperPenaltyWrapper` 与原项目同 |
| `config.py` | `EnvConfig` + `TrainConfig`,镜像 `task1_pick_banana` 的写法 |
| `collect_sim_demos.py` | Phase-1 演示数据采集(硬编码比例控制 + 噪声 + 可选 Octo embedding) |
| `collect_sim_classifier_data.py` | 奖励分类器数据采集(自动判定成功/失败,不需要人工标注) |
| `run_learner_conrft_pretrain.sh` | Phase-1 预训练启动脚本 |
| `run_learner_conrft.sh` | Phase-2 Learner 进程启动脚本 |
| `run_actor_conrft.sh` | Phase-2 Actor 进程启动脚本 |

新任务已经在 `examples/experiments/mappings.py` 中注册为 `task1_lift_cube_sim`。

---

## 3. 完整复现流程

所有命令的 **当前工作目录** 都是 `examples/experiments/task1_lift_cube_sim/`。

### Step 1 — 采集 Phase-1 演示数据 (~30 条成功轨迹)

```bash
python collect_sim_demos.py \
    --exp_name=task1_lift_cube_sim \
    --successes_needed=100 \
    --output=./demo_data/task1_lift_cube_sim_100_demos.pkl
```

如需带上 Octo embedding(与原 `record_demos_octo.py` 完全等价的输出格式):

```bash
python collect_sim_demos.py --with_octo=True ...
```

> 硬编码策略保证 100% 成功率,加入了高斯噪声(含旋转噪声)避免 BC 把
> `action[3:6]` 压到 0;详见脚本中 `compute_action()`。

### Step 2 — 采集奖励分类器数据

```bash
python collect_sim_classifier_data.py \
    --exp_name=task1_lift_cube_sim \
    --successes_needed=200
```

输出会落到 `./classifier_data/`,与原项目 `record_success_fail.py` 输出格式完全一致。

### Step 3 — 训练奖励分类器(无需修改原脚本)

```bash
python ../../train_reward_classifier.py --exp_name=task1_lift_cube_sim
```

训练完毕后会在当前目录下生成 `classifier_ckpt/`。

### Step 4 — Phase-1 离线预训练

```bash
bash run_learner_conrft_pretrain.sh
```

参数:`q_weight=0.1`,`bc_weight=1.0`,`pretrain_steps=20000`(BC 主导)。
预训练完成后 checkpoint 会保存在 `./conrft/`。

### Step 5 — Phase-2 在线训练(开两个终端)

终端 1(Learner):

```bash
bash run_learner_conrft.sh
```

终端 2(Actor):

```bash
bash run_actor_conrft.sh
```

参数:`q_weight=1.0`,`bc_weight=0.1`(RL 主导)。Actor 在仿真中执行策略采集数据,
Learner 从 buffer 训练并实时把参数推给 Actor。

### step 6——在仿真中测试阶段一效果保存视频命令

cd ~/sjl/conrft-main/examples/experiments/task1_lift_cube_sim

conda activate conrft

ulimit -n 65536

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

 

\# 开启 GPU 渲染（由于不需要跑 learner，评估时 JAX 可以直接使用 GPU）

export CUDA_VISIBLE_DEVICES=0

 

python ../../train_conrft_octo.py \

​    --exp_name=task1_lift_cube_sim \

​    --checkpoint_path=/home/robot/sjl/conrft-main/examples/experiments/task1_lift_cube_sim/conrft \

​    --eval_checkpoint_step=20000 \

​    --eval_n_trajs=20 \

​    --actor

---

## 4. 与真机版 (`task1_pick_banana`) 的差异说明

| 方面 | 真机版 | 仿真版 |
|---|---|---|
| 环境后端 | `FrankaEnv` → Flask → ROS → 真实 Franka | `SimFrankaEnv` → robosuite/MuJoCo |
| 数据采集 | 人用 SpaceMouse 操控 | 硬编码比例控制器 + 噪声 |
| 奖励分类器数据 | 人按空格标注成功 | 仿真中自动根据立方体高度判定 |
| 人类干预 | 有(SpaceMouse) | 无(Phase-2 完全自主探索) |
| 训练代码 | **不变** | **不变** |

> ConRFT 的 `train_conrft_octo.py`、`train_reward_classifier.py` 等核心训练代码
> 一行都没有改动,所有差异都封装在 `SimFrankaEnv` 内部。

---

## 5. 常见问题

**Q1. robosuite 安装报错 / 找不到 MuJoCo?**
A. 确保 Python ≥ 3.8,先 `pip install mujoco` 再装 `robosuite`。Linux 下 GPU 渲染需要正确配置
EGL,可参考 [robosuite 官方安装文档](https://robosuite.ai/docs/installation.html)。

**Q2. 采集脚本里 `cube_pos` 拿不到?**
A. 不同 robosuite 版本观测键名可能差异。已加兜底:`get_object_position()` 会
自动回退到 `sim.data.get_body_xpos("cube")`。如果都拿不到,在你装的 robosuite
版本中执行一次:

```python
import robosuite as suite
env = suite.make("Lift", robots="Panda", use_object_obs=True, has_offscreen_renderer=False, has_renderer=False, use_camera_obs=False)
print(env._get_observations().keys())
```

把里面真实的立方体键名(可能是 `cube_pos` / `Cube_pos` / `object_pos`)改回
`sim_env.py::get_object_position` 即可。

**Q3. Phase-2 收敛很慢?**
A. 仿真版没有 SpaceMouse 干预,完全靠策略自己探索。可以适当增大 `pretrain_steps`
让起点更好,或者在 Actor 中接入"脚本干预"(类似真机的 SpaceMouse 干预,但用
硬编码策略来代替人类),实现思路与 `franka_env.envs.wrappers.SpacemouseIntervention` 一致。

**Q4. 想换成更接近 pick-banana 的 pick-and-place 任务?**
A. 在 `wrapper.py::LiftCubeSimEnv.compute_reward` 中加上"放置目标位置"判定;
在 `sim_env.py::__init__` 中把 `env_name="Lift"` 换成 `env_name="PickPlace"`(注意
`PickPlace` 有 4 个物体和 bins,需要相应调整 `get_object_position` 的物体名)。
