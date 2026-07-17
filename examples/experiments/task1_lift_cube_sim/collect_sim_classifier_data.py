"""
仿真版奖励分类器数据采集脚本.

策略: 在仿真中我们能"作弊"地知道立方体高度,因此可以自动判定成功/失败,
不需要人手动按空格标注 (替代原 record_success_fail.py 的人工流程).

输出与原 record_success_fail.py 完全兼容:
  classifier_data/{exp_name}_{n}_success_images_{ts}.pkl
  classifier_data/{exp_name}_failure_images_{ts}.pkl
然后即可直接运行 train_reward_classifier.py.

用法:
    cd examples/experiments/task1_lift_cube_sim
    python experiments/task1_lift_cube_sim/collect_sim_classifier_data.py --exp_name=task1_lift_cube_sim \
        --successes_needed=200
"""
import copy
import datetime
import os
import sys
import pickle as pkl

import numpy as np
from absl import app, flags
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# /home/robot/sjl/conrft-main/examples/experiments/task1_lift_cube_sim/collect_sim_classifier_data.py

from experiments.mappings import CONFIG_MAPPING
from experiments.task1_lift_cube_sim.collect_sim_demos import (
    compute_action,
    reached,
    get_current_xyz,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "task1_lift_cube_sim", "")
flags.DEFINE_integer("successes_needed", 200, "Successful transitions to collect.")
flags.DEFINE_integer("max_episodes", 60, "Cap on episodes to run.")


def _make_transition(obs, action, next_obs, reward, done):
    return copy.deepcopy({
        "observations": obs,
        "actions": np.asarray(action, dtype=np.float32),
        "next_observations": next_obs,
        "rewards": float(reward),
        "masks": 1.0 - float(done),
        "dones": bool(done),
    })


def collect_episode(env, force_failure: bool = False):
    """
    跑一条 episode:
      - 正常模式: 硬编码完成 lift, 成功后保持立方体抬起 (后续帧都标 success)
      - 失败模式 (force_failure=True): 故意打乱动作, 几乎不会成功
    返回: (成功 transitions 列表, 失败 transitions 列表)
    """
    obs, info = env.reset()
    base_env = env.unwrapped
    cube_xyz = base_env.get_object_position("cube")

    successes = []
    failures = []

    if force_failure:
        # 在桌面上方随机游走, 不抓
        for _ in range(40):
            action = np.random.uniform(-1, 1, size=7).astype(np.float32)
            action[6] = 1.0  # 始终张开
            next_obs, reward, done, truncated, _ = env.step(action)
            failures.append(_make_transition(obs, action, next_obs, reward, done))
            obs = next_obs
            if done:
                break
        return successes, failures

    # 正常 lift 流程
    waypoints = [
        (cube_xyz + np.array([0.0, 0.0, 0.10]), 1.0, "approach"),
        (cube_xyz + np.array([0.0, 0.0, 0.005]), 1.0, "descend"),
        (cube_xyz + np.array([0.0, 0.0, 0.005]), -1.0, "close"),
        (cube_xyz + np.array([0.0, 0.0, 0.20]), -1.0, "lift"),
        (cube_xyz + np.array([0.0, 0.0, 0.20]), -1.0, "hold"),
    ]

    for target_xyz, gripper, phase in waypoints:
        if phase == "close":
            for _ in range(5):
                action = np.zeros(7, dtype=np.float32)
                action[6] = gripper
                next_obs, reward, done, truncated, _ = env.step(action)
                # close 阶段视觉上还没 lift, 标失败
                failures.append(_make_transition(obs, action, next_obs, reward, done))
                obs = next_obs
                if done:
                    return successes, failures
            continue

        if phase == "hold":
            # 保持抬起 5 步, 这些帧标成功 (用于训练分类器)
            for _ in range(5):
                action = np.zeros(7, dtype=np.float32)
                action[6] = -1.0
                next_obs, reward, done, truncated, _ = env.step(action)
                if reward > 0:
                    successes.append(_make_transition(obs, action, next_obs, reward, done))
                else:
                    failures.append(_make_transition(obs, action, next_obs, reward, done))
                obs = next_obs
                if done:
                    return successes, failures
            continue

        for _ in range(40):
            current_xyz = get_current_xyz(base_env)
            if reached(current_xyz, target_xyz, threshold=0.015):
                break
            action = compute_action(
                current_xyz, target_xyz, gripper,
                action_scale_xyz=base_env.action_scale[0],
                noise_xyz=0.02, noise_rpy=0.02,
            )
            next_obs, reward, done, truncated, _ = env.step(action)
            if reward > 0:
                successes.append(_make_transition(obs, action, next_obs, reward, done))
            else:
                failures.append(_make_transition(obs, action, next_obs, reward, done))
            obs = next_obs
            if done:
                return successes, failures

    return successes, failures


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, f"{FLAGS.exp_name} 未注册"
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    successes_total = []
    failures_total = []
    pbar = tqdm(total=FLAGS.successes_needed)

    ep = 0
    while len(successes_total) < FLAGS.successes_needed and ep < FLAGS.max_episodes:
        # 每 5 条 episode 中安排 1 条故意失败, 平衡正负样本视觉多样性
        force_fail = (ep % 5 == 4)
        s, f = collect_episode(env, force_failure=force_fail)
        n_added = min(len(s), FLAGS.successes_needed - len(successes_total))
        successes_total.extend(s[:n_added])
        pbar.update(n_added)
        failures_total.extend(f)
        ep += 1
        pbar.set_description(
            f"ep={ep} success={len(successes_total)} failure={len(failures_total)}")

    pbar.close()
    env.close()

    os.makedirs("./classifier_data", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    succ_path = f"./classifier_data/{FLAGS.exp_name}_{len(successes_total)}_success_images_{ts}.pkl"
    fail_path = f"./classifier_data/{FLAGS.exp_name}_failure_images_{ts}.pkl"
    with open(succ_path, "wb") as f:
        pkl.dump(successes_total, f)
    with open(fail_path, "wb") as f:
        pkl.dump(failures_total, f)
    print(f"[OK] 成功样本 {len(successes_total)} -> {succ_path}")
    print(f"[OK] 失败样本 {len(failures_total)} -> {fail_path}")


if __name__ == "__main__":
    app.run(main)
