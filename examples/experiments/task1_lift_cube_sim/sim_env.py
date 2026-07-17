"""
SimFrankaEnv: 基于 robosuite 的仿真环境，接口完全模仿 FrankaEnv，
使得所有训练代码、wrapper 都可以直接使用而无需修改。

设计原则:
1. action_space / observation_space 与 FrankaEnv 完全一致
2. 暴露 currpos / curr_gripper_pos / currvel 等属性，兼容 PickBananaEnv 子类
3. 支持 fake_env=True，用于训练奖励分类器时只需 obs_space 定义的场景
4. 内部维护一个 robosuite Lift 环境，将 7 维 action 转发到 robosuite 的 OSC_POSE 控制器
"""
import time
from collections import OrderedDict

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    ROBOSUITE_AVAILABLE = True
except ImportError:
    ROBOSUITE_AVAILABLE = False

from franka_env.envs.franka_env import DefaultEnvConfig


class SimFrankaEnv(gym.Env):
    """
    仿真版 FrankaEnv。底层使用 robosuite (MuJoCo) 而非真机 + ROS + Flask。

    关键映射 (robosuite -> FrankaEnv):
      - robot0_eef_pos + robot0_eef_quat       -> state.tcp_pose (7,)
      - robot0_gripper_qpos (归一化)           -> state.gripper_pose (1,)
      - agentview_image                        -> images.side_policy_256, side_classifier
      - robot0_eye_in_hand_image               -> images.wrist_1
    """

    def __init__(self, hz: int = 10, fake_env: bool = False,
                 save_video: bool = False, config: DefaultEnvConfig = None,
                 set_load: bool = False):
        self.config = config
        self.hz = hz
        self.action_scale = np.array(config.ACTION_SCALE, dtype=np.float32)
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.curr_path_length = 0

        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self._RESET_POSE = np.array(config.RESET_POSE, dtype=np.float32)

        # ----- 1) 定义 action / observation 空间 (与 FrankaEnv 完全一致) -----
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        img_h_wrist, img_w_wrist = 128, 128
        img_h_side, img_w_side = 256, 256
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose":     gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel":      gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "tcp_force":    gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "tcp_torque":   gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "gripper_pose": gym.spaces.Box(-1.0, 1.0,    shape=(1,), dtype=np.float32),
            }),
            "images": gym.spaces.Dict({
                "wrist_1":          gym.spaces.Box(0, 255, shape=(img_h_wrist, img_w_wrist, 3), dtype=np.uint8),
                "side_policy_256":  gym.spaces.Box(0, 255, shape=(img_h_side, img_w_side, 3), dtype=np.uint8),
                "side_classifier":  gym.spaces.Box(0, 255, shape=(img_h_side, img_w_side, 3), dtype=np.uint8),
            }),
        })

        # ----- 2) 内部状态缓存 (兼容 FrankaEnv 子类访问的属性) -----
        self.currpos = np.zeros(7, dtype=np.float32)
        self.currpos[6] = 1.0  # 默认四元数 w=1
        self.currvel = np.zeros(6, dtype=np.float32)
        self.currforce = np.zeros(3, dtype=np.float32)
        self.currtorque = np.zeros(3, dtype=np.float32)
        self.curr_gripper_pos = np.array([1.0], dtype=np.float32)
        self.q = np.zeros(7, dtype=np.float32)
        self.dq = np.zeros(7, dtype=np.float32)
        self.currjacobian = np.zeros((6, 7), dtype=np.float32)
        self.terminate = False
        self._cached_images = {
            "wrist_1":         np.zeros((img_h_wrist, img_w_wrist, 3), dtype=np.uint8),
            "side_policy_256": np.zeros((img_h_side, img_w_side, 3), dtype=np.uint8),
            "side_classifier": np.zeros((img_h_side, img_w_side, 3), dtype=np.uint8),
        }
        self._last_robosuite_obs = None

        # —— 稠密奖励权重(阶段门控:接近->抓握->抬升->成功) ——
        self.w_reach = 1.0           # 接近项: 夹爪->物体 的负距离权重
        self.grasp_bonus = 1.0       # 抓住物体的基础分
        self.w_lift = 5.0            # 抬升项: 物体->目标高度 的负距离权重
        self.success_bonus = 10.0    # 成功大奖励(要压倒其它项)
        self.target_height = 0.86    # Lift 的目标高度(沿用原成功线)
        self.success_hold_steps = 5  # 连续保持成功多少步才算真成功(防"抬一下就掉")
        self._success_hold = 0       # 成功保持计数器(运行时用)

        # fake_env 仅返回空间定义,用于训练奖励分类器
        if fake_env:
            self.sim_env = None
            return

        if not ROBOSUITE_AVAILABLE:
            raise ImportError(
                "robosuite 未安装。请先执行: pip install robosuite"
            )

        # ----- 3) 创建 robosuite 仿真环境 -----
        controller_config = load_controller_config(default_controller="OSC_POSE")
        # 让控制器输出范围与 ConRFT 的 action_scale 对齐:
        # action[:3] in [-1,1] -> 位移最大 ±action_scale[0] 米
        # action[3:6] in [-1,1] -> 旋转最大 ±action_scale[1] 弧度
        controller_config["output_max"] = [
            float(self.action_scale[0])] * 3 + [float(self.action_scale[1])] * 3
        controller_config["output_min"] = [
            -float(self.action_scale[0])] * 3 + [-float(self.action_scale[1])] * 3
        controller_config["input_max"] = 1.0
        controller_config["input_min"] = -1.0
        controller_config["control_delta"] = True
        controller_config["impedance_mode"] = "fixed"
        controller_config["uncouple_pos_ori"] = True

        self.sim_env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=save_video,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[img_h_side, img_h_wrist],
            camera_widths=[img_w_side, img_w_wrist],
            control_freq=hz,
            horizon=10000,
            ignore_done=True,
            use_object_obs=True,
            reward_shaping=False,
        )

    # ---------------------------------------------------------------------
    # gym 标准接口：这个reset函数没有实现课程学习，后面那个rest函数实现了逆向课程学习
    # ---------------------------------------------------------------------
    '''
    def reset(self, **kwargs):
        if self.sim_env is None:
            raise RuntimeError("fake_env=True 时不能调用 reset()")

        self.sim_env.reset()
        self.curr_path_length = 0
        self.terminate = False

        # 可选: 从默认初始位姿出发先做一些扰动,模拟真机 RANDOM_RESET
        if self.randomreset:
            self._apply_random_reset_perturbation()

        obs_dict = self.sim_env._get_observations(force_update=True)
        self._update_state_from_robosuite(obs_dict)
        self._last_robosuite_obs = obs_dict
        return self._get_obs(), {"succeed": False}
        '''
    def reset(self, seed=None, options=None, **kwargs):
        if self.sim_env is None:
            raise RuntimeError("fake_env=True 时不能调用 reset()")
 
        self.sim_env.reset()
        self.curr_path_length = 0
        self._success_hold = 0
        self.terminate = False

        # reset 后立刻刷新一次观测，保证课程脚本拿到本局方块的真实位置，如果这里步刷新那么拿到的方块位置就是上一次回合结束的位置
        obs_dict = self.sim_env._get_observations(force_update=True)
        self._update_state_from_robosuite(obs_dict)
        self._last_robosuite_obs = obs_dict
 
        # 从 Gymnasium 的 options 字典中提取相对步数，若没有提供则默认设为 999999（执行标准任务）
        current_step = 999999
        if options is not None and "current_step" in options:
            current_step = options["current_step"]
 
        # 执行逆向课程逻辑
        if current_step < 10000:
            # 【第一关：1万步之前】一出生夹爪已经把方块捏死在桌面上（只需学抬起）
            self._move_to_cube_script(close_gripper=True,
                              xy_range=0.003, z_range=0.0,
                              min_steps=12, max_steps=13)
        elif current_step < 20000:
            # 【第二关：1万~2万步】一出生夹爪套在方块外侧但处于张开状态（学合拢+抬起）
            self._move_to_cube_script(close_gripper=False,
                              xy_range=0.01, z_range=0,
                              min_steps=12, max_steps=14)
        else:
            # 【第三关：2万步以后】标准重置状态（学下落+合拢+抬起）
            if self.randomreset:
                self._apply_random_reset_perturbation()

        obs_dict = self.sim_env._get_observations(force_update=True)
        self._update_state_from_robosuite(obs_dict)
        self._last_robosuite_obs = obs_dict
        return self._get_obs(), {"succeed": False}

        

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 注意: ConRFT 中 action[3:6] 是 xyz 欧拉角增量;
        # robosuite OSC_POSE 输入的是轴角向量 (rotation vector)。
        # 在小角度下两者数值近似相等 (本任务 action_scale=0.2 rad 足够小),
        # 因此可以直接转发。如需严格对齐, 可在此处做欧拉->轴角转换。
        sim_action = np.zeros(7, dtype=np.float32)
        sim_action[:6] = action[:6]
        # robosuite 夹爪约定: -1=张开, 1=闭合; ConRFT 约定: 1=张开, -1=闭合
        sim_action[6] = -action[6]

        obs_dict, _, _, _ = self.sim_env.step(sim_action)
        self._last_robosuite_obs = obs_dict
        self._update_state_from_robosuite(obs_dict)

        self.curr_path_length += 1
        reward = self.compute_reward(self._get_obs())   # 浮点稠密奖励,不要 int()

        # 一旦满足成功条件即视为成功(compute_reward 中已置 self.success=True)
        success_done = getattr(self, "success", False)

        done = (self.curr_path_length >= self.max_episode_length
                or success_done or self.terminate)

        # 维持 hz 节奏 (仿真本身已按 control_freq 推进, 这里仅占位)
        dt = time.time() - start_time
        if dt < (1.0 / self.hz):
            pass  # 不强制 sleep, 加速仿真训练

        return self._get_obs(), reward, done, False, {"succeed": success_done}

    # ---------------------------------------------------------------------------------------- #
    # 下面这个奖励函数定义的是稀疏奖励，整个episode成功才有1的奖励，否则无奖励，后面的奖励是稠密奖励
    # ---------------------------------------------------------------------------------------- #
    '''
    def compute_reward(self, obs) -> int:
        """子类应当 override。这里给一个默认的 Lift 任务奖励:
        立方体 z 高于初始 + 0.04 即视为成功。"""
        if self._last_robosuite_obs is None:
            return 0
        cube_z = float(self._last_robosuite_obs.get("cube_pos", [0, 0, 0])[2])
        # robosuite Lift 桌面约 0.82, 抓起后 cube_z >= 0.86 视为成功
        return int(cube_z > 0.86)
    '''

    def _is_grasping(self) -> bool:
        """检测是否真正抓住物体;robosuite 接口失败时用'夹爪贴近物体'兜底。"""
        try:
            return bool(self.sim_env._check_grasp(
                gripper=self.sim_env.robots[0].gripper,
                object_geoms=self.sim_env.cube))
        except Exception:
            if self._last_robosuite_obs is None:
                return False
            cube = np.array(self._last_robosuite_obs.get("cube_pos", [0, 0, 0]), dtype=np.float32)
            grip = np.array(self._last_robosuite_obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float32)
            return float(np.linalg.norm(grip - cube)) < 0.02

    def compute_reward(self, obs) -> float:
        """阶段门控稠密奖励(为迁移 pick-and-place 设计, goal=target_height 可替换)。
        阶段1(未抓住): 奖励靠近物体; 阶段2(已抓住): 奖励把物体抬到目标高度。"""
        if self._last_robosuite_obs is None:
            return 0.0
        cube = np.array(self._last_robosuite_obs.get("cube_pos", [0, 0, 0]), dtype=np.float32)
        grip = np.array(self._last_robosuite_obs.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float32)
        grasped = self._is_grasping()

        if not grasped:
            # 阶段1: 靠近物体(也负责掉落后把夹爪拉回去)
            d_reach = float(np.linalg.norm(grip - cube))
            r = self.w_reach * (-d_reach)
        else:
            # 阶段2: 已抓住 -> 抬升到目标高度
            r = self.grasp_bonus
            height_gap = max(0.0, self.target_height - float(cube[2]))
            r += self.w_lift * (-height_gap)

        # 成功大奖励: 抓住 且 已到达目标高度
        if grasped and float(cube[2]) > self.target_height:
            r += self.success_bonus
            self._last_success = True
        else:
            self._last_success = False
        return float(r)



    def close(self):
        if self.sim_env is not None:
            try:
                self.sim_env.close()
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # 子类/外部脚本会用到的辅助方法
    # ---------------------------------------------------------------------
    def get_object_position(self, name: str = "cube") -> np.ndarray:
        """从仿真中读取目标物体位置 (硬编码策略采集时使用)。"""
        if self._last_robosuite_obs is None:
            return np.zeros(3)
        key = f"{name}_pos"
        if key in self._last_robosuite_obs:
            return np.array(self._last_robosuite_obs[key], dtype=np.float32)
        # 兜底: 直接通过 mujoco 模型查询
        try:
            sim = self.sim_env.sim
            return np.array(sim.data.get_body_xpos(name), dtype=np.float32)
        except Exception:
            return np.zeros(3)

    def get_target_position(self) -> np.ndarray:
        """返回 "放置目标" 的虚拟位置 (本任务暂用 cube 起始位置上方)。"""
        cube_pos = self.get_object_position("cube")
        return cube_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)

    # ---------------------------------------------------------------------
    # 内部辅助：
    # reset函数会让臂和目标立马回到初始位置，但在物理世界会有惯性，导致臂抖动
    # 这个函数就是避免这些抖动，达到稳定状态的
    # ---------------------------------------------------------------------
    def _apply_random_reset_perturbation(self):
        """在 robosuite 默认 reset 后,通过执行若干步零动作进入稳态。
        若需要随机 xy 抖动, 可在此处添加策略 (这里依赖 robosuite 自带的物体位置随机化)。"""
        for _ in range(2):
            zero_act = np.zeros(7, dtype=np.float32)
            self.sim_env.step(zero_act)
        

    # ---------------------------------------------------------------------
    # 内部辅助:逆向课程学习控制函数
    # 这个函数的原理是在课程学习之前的这12步也是策略自己走的，但是进行课程学习之后
    # reset始终会回到初始位置，所以就需要用控制器走这若干步让机械臂接近/抓住物体，
    # 至于为什么是12步，这是最大冗余步，也可以是13、14、15步
    #
    # 为实现"由单个初始状态 -> 一个初始状态分布"，新增以下随机扰动参数:
    #   xy_range  : 目标抓取点在水平(x,y)方向的随机偏移幅度(米)
    #   z_range   : 目标抓取点在竖直(z)方向向上的随机偏移幅度(米)
    #   min_steps : 接近过程的最少步数
    #   max_steps : 接近过程的最多步数(在[min_steps, max_steps]内随机)
    # 课程1扰动应最小(保证抓取稳固)，课程2适当放大，符合逆向课程"越靠近目标越窄"的原则
    # ---------------------------------------------------------------------
    def _move_to_cube_script(self, close_gripper=False, xy_range=0.0, z_range=0.0,
                             min_steps=12, max_steps=12):
        """利用底层控制器，自动将夹爪移动到红方块附近（可选择合拢夹爪）。
        通过 xy_range/z_range/min_steps/max_steps 给初始状态加随机扰动，
        使每局的起始状态各不相同，从而由单点变成一个初始状态分布。"""
        cube_pos = self.get_object_position("cube")

        # —— 在方块位置上加随机扰动，得到本局的目标抓取点(起始态分布的来源) ——
        offset = np.array([
            np.random.uniform(-xy_range, xy_range),   # x 随机偏移
            np.random.uniform(-xy_range, xy_range),   # y 随机偏移
            np.random.uniform(0.0, z_range),          # z 只向上偏(模拟"偏高/够不到")
        ], dtype=np.float32)
        target_pos = cube_pos + offset

        # —— 接近步数也随机(范围由关卡控制，课程1收紧以避免还没到位就闭合导致抓空) ——
        n_steps = np.random.randint(min_steps, max_steps + 1)

        # 1. 快速下落并对准(带扰动的)目标抓取点
        for _ in range(n_steps):  # 运行 n_steps 步快速移动过去
            curr_pos = self.currpos[:3]
            pos_diff = target_pos - curr_pos          # 注意:对准的是 target_pos(含扰动)
            # 为了防止直接撞飞方块，Z 轴留出 1.5 厘米的安全抓取高度
            #pos_diff[2] += 0.015

            action = np.zeros(7, dtype=np.float32)
            # 计算平移位移并缩放到动作范围 [-1, 1]
            action[:3] = np.clip(pos_diff / self.action_scale[0], -1.0, 1.0)
            action[6] = 1.0  # 接近过程中始终张开夹爪，不然闭合的话会撞飞目标

            # 转发至仿真器并刷新内部状态
            sim_action = np.zeros(7, dtype=np.float32)
            sim_action[:6] = action[:6]
            sim_action[6] = -action[6]
            self.sim_env.step(sim_action)
            self._update_state_from_robosuite(self.sim_env._get_observations(force_update=True))

        # 2. 如果是第一关，额外执行几次完全捏紧动作
        if close_gripper:
            for _ in range(8):
                action = np.zeros(7, dtype=np.float32)
                action[6] = -1.0  # 闭合夹爪
                sim_action = np.zeros(7, dtype=np.float32)
                sim_action[6] = -action[6]
                self.sim_env.step(sim_action)
                self._update_state_from_robosuite(self.sim_env._get_observations(force_update=True))



    def _update_state_from_robosuite(self, obs_dict: dict):
        eef_pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)
        # robosuite 四元数已是 [x, y, z, w] (与 scipy 一致)
        self.currpos = np.concatenate([eef_pos, eef_quat]).astype(np.float32)

        # robosuite 没有直接给 6D 末端速度,简单置零 (训练中此项作为本体感受,影响较小)
        self.currvel = np.zeros(6, dtype=np.float32)
        self.currforce = np.zeros(3, dtype=np.float32)
        self.currtorque = np.zeros(3, dtype=np.float32)

        # 关节状态
        self.q = np.asarray(obs_dict.get("robot0_joint_pos", np.zeros(7)),
                            dtype=np.float32)[:7]
        self.dq = np.asarray(obs_dict.get("robot0_joint_vel", np.zeros(7)),
                             dtype=np.float32)[:7]

        # 夹爪位置: Panda gripper qpos 两个手指符号相反 [+x, -x]
        # 开合宽度 = qpos[0] - qpos[1], 全闭≈0, 全开≈0.08
        gripper_qpos = np.asarray(
            obs_dict.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32)
        gripper_width = float(gripper_qpos[0] - gripper_qpos[1])
        max_width = 0.08  # Panda 夹爪最大开口
        # 归一化到 [-1, 1]: 0宽度=完全闭合(-1), max_width=完全张开(+1)
        gripper_open_norm = np.clip(gripper_width / max_width, 0.0, 1.0) * 2 - 1
        self.curr_gripper_pos = np.array([gripper_open_norm], dtype=np.float32)

        # 图像: robosuite 默认返回 RGB 上下颠倒, 这里翻转修正
        wrist_img = obs_dict.get("robot0_eye_in_hand_image")
        side_img = obs_dict.get("agentview_image")
        if wrist_img is not None:
            self._cached_images["wrist_1"] = np.ascontiguousarray(wrist_img[::-1])
        if side_img is not None:
            self._cached_images["side_policy_256"] = np.ascontiguousarray(side_img[::-1])
            self._cached_images["side_classifier"] = self._cached_images["side_policy_256"]

    def _get_obs(self) -> dict:
        return {
            "state": {
                "tcp_pose":     self.currpos.astype(np.float32),
                "tcp_vel":      self.currvel.astype(np.float32),
                "tcp_force":    self.currforce.astype(np.float32),
                "tcp_torque":   self.currtorque.astype(np.float32),
                "gripper_pose": self.curr_gripper_pos.astype(np.float32),
            },
            "images": {
                "wrist_1":         self._cached_images["wrist_1"],
                "side_policy_256": self._cached_images["side_policy_256"],
                "side_classifier": self._cached_images["side_classifier"],
            },
        }
