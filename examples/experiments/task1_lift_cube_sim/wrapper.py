"""
LiftCubeSimEnv: 在 SimFrankaEnv 之上,定义具体任务的成功条件和 reset 行为。
GripperPenaltyWrapper: 与原 task1_pick_banana 中完全一致,处理夹爪开合惩罚。
"""
import copy
from typing import OrderedDict

import gymnasium as gym
import numpy as np

from experiments.task1_lift_cube_sim.sim_env import SimFrankaEnv


class LiftCubeSimEnv(SimFrankaEnv):
    """
    简化的 "抓起立方体" 任务 (作为 pick-banana 的仿真代理任务)。
    成功条件: 立方体被抓起且高度超过阈值。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cube_init_z = None  # 记录 reset 时立方体的初始 z

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # 记录立方体初始高度
        if self._last_robosuite_obs is not None and "cube_pos" in self._last_robosuite_obs:
            self._cube_init_z = float(self._last_robosuite_obs["cube_pos"][2])
        else:
            self._cube_init_z = 0.82  # robosuite Lift 桌面高度兜底
        self.success = False
        return obs, info

    def compute_reward(self, obs) -> int:
        """成功 = 立方体被抓起 (高于初始高度 0.04m) 并且夹爪闭合。"""
        if self._last_robosuite_obs is None:
            return 0
        cube_pos = self._last_robosuite_obs.get("cube_pos")
        if cube_pos is None:
            return 0
        cube_z = float(cube_pos[2])

        # 条件 1: 立方体被抬起
        lifted = cube_z > (self._cube_init_z + 0.04)
        # 条件 2: 夹爪处于闭合状态 (gripper_pose 接近 -1)
        gripper_closed = float(self.curr_gripper_pos[0]) < 0.2

        if lifted and gripper_closed:
            self.success = True
            #self._last_success = True   # ← 加这行
            return 1
        #self._last_success = False      # ← 加这行
        return 0


class GripperPenaltyWrapper(gym.Wrapper):
    """与 task1_pick_banana 中的 wrapper 完全一致, 处理夹爪开合惩罚。"""

    def __init__(self, env, penalty: float = -0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        action = copy.deepcopy(action)
        grasp_action = action[..., -1]
        grasp_action = np.where(grasp_action > 0.5, 1,
                                np.where(grasp_action < -0.5, -1, 0))
        action[..., -1] = grasp_action

        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.7) or (
                action[-1] > 0.5 and self.last_gripper_pos < 0.7):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
