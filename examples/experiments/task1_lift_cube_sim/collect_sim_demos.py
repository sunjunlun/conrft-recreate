"""
仿真版 Phase-1 演示数据采集脚本。

策略: 硬编码比例控制器 + 高斯噪声 (含旋转噪声) + Octo embedding 后处理.
  - 硬编码保证 100% 成功率, 避免在仿真中跑随机策略浪费时间
  - 噪声让 action[3:6] 不再恒为 0, 给 Phase-1 BC 提供旋转维度的多样性
  - 用 Octo 模型只算 embedding, 不做决策, 与原 record_demos_octo.py 输出格式一致

使用:
    cd examples/experiments/task1_lift_cube_sim
    python experiments/task1_lift_cube_sim/collect_sim_demos.py --exp_name=task1_lift_cube_sim --successes_needed=30 --output=./demo_data/task1_lift_cube_sim_30_demos.pkl
"""
import copy
import datetime
import os
import sys
import pickle as pkl
import time

import numpy as np
from absl import app, flags
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.mappings import CONFIG_MAPPING
from data_util import (
    add_mc_returns_to_trajectory,
    add_embeddings_to_trajectory,
    add_next_embeddings_to_trajectory,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task1_lift_cube_sim", "Experiment name.")
flags.DEFINE_integer("successes_needed", 30, "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "")
flags.DEFINE_float("reward_bias", 0.0, "")
flags.DEFINE_string("output", "./demo_data/task1_lift_cube_sim_30_demos.pkl", "Output pkl path.")
flags.DEFINE_boolean("with_octo", False,
                     "If True, also compute Octo embeddings (slower, but matches original pipeline).")
flags.DEFINE_float("noise_xyz", 0.05, "Std dev of gaussian noise added to action[:3].")
flags.DEFINE_float("noise_rpy", 0.05, "Std dev of gaussian noise added to action[3:6].")


# =====================================================================
# 硬编码策略
# =====================================================================
def compute_action(current_xyz, target_xyz, gripper, action_scale_xyz=0.05,
                   noise_xyz=0.05, noise_rpy=0.05):
    """
    一步比例控制: action = (target - current) / scale, 加少量噪声.
    旋转部分输出小的随机噪声 (而非 0), 避免 Phase-1 BC 把旋转输出压到 0.
    """
    delta_xyz = np.asarray(target_xyz, dtype=np.float32) - np.asarray(current_xyz, dtype=np.float32)
    action_xyz = np.clip(delta_xyz / action_scale_xyz, -1.0, 1.0)

    # 位移噪声
    if noise_xyz > 0:
        action_xyz = np.clip(action_xyz + np.random.normal(0, noise_xyz, size=3), -1.0, 1.0)

    # 旋转部分: 只用噪声 (本任务不需要主动旋转)
    if noise_rpy > 0:
        action_rpy = np.clip(np.random.normal(0, noise_rpy, size=3), -1.0, 1.0)
    else:
        action_rpy = np.zeros(3, dtype=np.float32)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = action_xyz
    action[3:6] = action_rpy
    action[6] = gripper
    return action.astype(np.float32)


def reached(current_xyz, target_xyz, threshold=0.02):
    return float(np.linalg.norm(np.asarray(target_xyz) - np.asarray(current_xyz))) < threshold


def world_to_ee_frame(action, base_env):
    """将世界坐标系的 action[:6] 转换到末端坐标系。
    RelativeFrame wrapper 期望输入在末端坐标系,会用伴随矩阵变换回世界坐标系。
    因此硬编码控制器算出世界系 delta 后,需要先乘以 R_ee^{-1} 才能正确执行。"""
    quat = base_env.currpos[3:7]  # (qx, qy, qz, qw)
    R_ee = Rotation.from_quat(quat).as_matrix()
    R_inv = R_ee.T  # 旋转矩阵正交, 逆=转置
    action = action.copy()
    action[:3] = R_inv @ action[:3]
    action[3:6] = R_inv @ action[3:6]
    return action


def get_current_xyz(env):
    """从环境中获取当前末端位置 (世界坐标系)。"""
    return np.asarray(env.unwrapped.currpos[:3], dtype=np.float32)


def collect_one_episode(env, max_steps_per_phase=40):
    """
    硬编码执行一次完整的 lift 流程,返回 (成功标志, 轨迹列表).
    轨迹中每个元素是一个 dict: {observations, actions, next_observations, rewards, masks, dones, infos}
    """
    obs, info = env.reset()
    base_env = env.unwrapped
    cube_xyz = base_env.get_object_position("cube")

    # -------------------------------------------------问题：物体可能不在机械臂的工作范围之内---------------------------------------- #
    # 初始时机械臂是被设置在任意位置的，这里的waypoints是4个目标点，也即是第一阶段先到达cube_xyz + np.array([0.0, 0.0, 0.10])
    # 第二阶段在到达cube_xyz + np.array([0.0, 0.0, 0.005])，以此类推。
    # ------------------------------------------------问题：物体可能不在机械臂的工作范围之内----------------------------------------- #
    waypoints = [
        # (目标 xyz, gripper, 描述)
        (cube_xyz + np.array([0.0, 0.0, 0.10]),  1.0, "above_cube"),
        (cube_xyz + np.array([0.0, 0.0, 0.005]), 1.0, "descend"),
        (cube_xyz + np.array([0.0, 0.0, 0.005]), -1.0, "close_gripper"),
        (cube_xyz + np.array([0.0, 0.0, 0.20]),  -1.0, "lift"),
    ]

    trajectory = []
    succeeded = False
    done = False

    for target_xyz, gripper, _phase in waypoints:
        if done:
            break
        # close_gripper 阶段只发几步全停止动作让夹爪闭合到位
        if _phase == "close_gripper":
            for _ in range(5):
                action = np.zeros(7, dtype=np.float32)
                action[6] = gripper
                action += np.random.normal(0, 0.01, size=7).astype(np.float32) * \
                    np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
                action_ee = world_to_ee_frame(action, base_env)
                next_obs, reward, done, truncated, step_info = env.step(action_ee)
                trajectory.append(_make_transition(obs, action_ee, next_obs, reward,
                                                    done, step_info))
                obs = next_obs
                if reward > 0:
                    succeeded = True
                if done:
                    break
            continue

        # 普通阶段: 比例控制移动
        is_lift = (_phase == "lift")          # 新增:标记是否为抬升阶段
        for _ in range(max_steps_per_phase):
            current_xyz = get_current_xyz(base_env)
            if reached(current_xyz, target_xyz, threshold=0.015):
                break
            action = compute_action(
                current_xyz, target_xyz, gripper,
                action_scale_xyz=base_env.action_scale[0],
                noise_xyz=FLAGS.noise_xyz,
                noise_rpy=FLAGS.noise_rpy,
            )
            action_ee = world_to_ee_frame(action, base_env)
            next_obs, reward, done, truncated, step_info = env.step(action_ee)
            trajectory.append(_make_transition(obs, action_ee, next_obs, reward,
                                                done, step_info))
            obs = next_obs
            if reward > 0:
                succeeded = True
            if done and not is_lift:          # 改:lift 阶段无视 done,继续爬到 20cm
                break

    return succeeded, trajectory


def _make_transition(obs, action, next_obs, reward, done, info):
    return copy.deepcopy({
        "observations": obs,
        "actions": np.asarray(action, dtype=np.float32),
        "next_observations": next_obs,
        "rewards": float(reward),
        "masks": 1.0 - float(done),
        "dones": bool(done),
        "infos": info,
    })


# =====================================================================
# 主流程
# =====================================================================
def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, \
        f"Experiment {FLAGS.exp_name} 未注册, 请先在 mappings.py 中添加。"
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # 用真实环境 (非 fake), 但不使用奖励分类器 (此时尚未训练好)
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    # ----- 可选: 加载 Octo 模型用于补 embedding -----
    octo_model = None
    octo_tasks = None
    if FLAGS.with_octo:
        try:
            from octo.model.octo_model import OctoModel
            octo_model = OctoModel.load_pretrained(config.octo_path)
            octo_tasks = octo_model.create_tasks(texts=[config.task_desc])
            print(f"[INFO] Octo 模型已加载: {config.octo_path}")
        except Exception as e:
            print(f"[WARN] 加载 Octo 失败 ({e}), 将不计算 embedding。")
            octo_model = None

    success_count = 0
    fail_count = 0
    all_transitions = []
    pbar = tqdm(total=FLAGS.successes_needed)
    t0 = time.time()

    while success_count < FLAGS.successes_needed:
        succeeded, trajectory = collect_one_episode(env)
        if succeeded and len(trajectory) > 0:
            # 后处理 (与 record_demos_octo.py 完全一致)
            trajectory = add_mc_returns_to_trajectory(
                trajectory, config.discount, FLAGS.reward_scale,
                FLAGS.reward_bias, config.reward_neg, is_sparse_reward=True)
            if octo_model is not None:
                trajectory = add_embeddings_to_trajectory(
                    trajectory, octo_model, tasks=octo_tasks)
                trajectory = add_next_embeddings_to_trajectory(trajectory)
            all_transitions.extend(trajectory)
            success_count += 1
            pbar.update(1)
        else:
            fail_count += 1
        pbar.set_description(
            f"成功 {success_count}/{FLAGS.successes_needed}, 失败 {fail_count}, "
            f"用时 {time.time() - t0:.0f}s")

    pbar.close()
    env.close()

    out_dir = os.path.dirname(FLAGS.output) or "./demo_data"
    os.makedirs(out_dir, exist_ok=True)
    with open(FLAGS.output, "wb") as f:
        pkl.dump(all_transitions, f)
    print(f"[OK] 已保存 {success_count} 条成功轨迹 ({len(all_transitions)} 条 transitions) -> {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
