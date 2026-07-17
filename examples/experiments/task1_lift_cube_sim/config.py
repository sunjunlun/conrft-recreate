"""
任务 task1_lift_cube_sim 的配置: 仿真版 (robosuite/MuJoCo).
镜像 task1_pick_banana/config.py 的结构,主要改动:
  1) 环境换成 LiftCubeSimEnv (基于 SimFrankaEnv)
  2) 移除 SpacemouseIntervention (仿真中无 SpaceMouse)
  3) 相机 / 裁剪等只保留接口需要的部分
"""
import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.task1_lift_cube_sim.wrapper import (
    LiftCubeSimEnv,
    GripperPenaltyWrapper,
)


class EnvConfig(DefaultEnvConfig):
    # 仿真不需要 SERVER_URL / 真实相机序列号, 但保留这些字段以兼容父类
    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = { # 定义了三个视角的相机画面
        "wrist_1":         {"serial_number": "sim", "dim": (128, 128), "exposure": 0},
        "side_policy_256": {"serial_number": "sim", "dim": (256, 256), "exposure": 0},
        "side_classifier": {"serial_number": "sim", "dim": (128, 128), "exposure": 0},
    }
    IMAGE_CROP = { # 分别对三个视角相机的图像裁剪，这里没有裁剪直接使用的
        "wrist_1":         lambda img: img,
        "side_policy_256": lambda img: img,
        "side_classifier": lambda img: img,
    }

    # 仿真坐标系下的近似 reset / target 位姿 (robosuite 默认 Panda 在 (0,0,0)
    # 处, 桌面 z≈0.82, 立方体在桌中心附近)
    TARGET_POSE = np.array([0.0, 0.0, 0.90, np.pi, 0, 0])
    RESET_POSE = np.array([0.0, 0.0, 1.05, np.pi, 0, 0])
    ACTION_SCALE = np.array([0.05, 0.2, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = False # 是否在运行过程中使用 cv2.imshow 实时在屏幕上弹出相机画面窗口。在后台服务器训练时设为 False 以免报错。
    RANDOM_XY_RANGE = 0.02 # 方块在 X 和 Y 轴上最多偏移 2 厘米
    RANDOM_RZ_RANGE = 0.03 # 方块在偏航角（Z轴旋转）上最多随机旋转 0.03 弧度（约 1.7°）
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.3, 0.20, 0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.3, 0.3, 0.10, 0.5, 0.5, 0.5])
    # COMPLIANCE_PARAM / PRECISION_PARAM 仿真中不会用到, 保留空字典占位
    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}
    MAX_EPISODE_LENGTH = 150


class TrainConfig(DefaultTrainingConfig):
    batch_size = 32  # 默认256，减半以适应16GB GPU双进程
    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = ["side_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2   # 这里设为 2，表示每训练 2 次 Critic 权重，才更新 1 次 Actor 策略权重
    random_steps = 0 # 在线训练开始前，Actor 采取纯随机动作收集数据的步数。因为我们有阶段一 BC 预训练基础，因此设为 0，一开局就用策略网络探索
    discount = 0.98
    buffer_period = 1000     # 这个变量表示buffer每隔多少步保存一次，0表示不保存
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Pick up the red cube"
    replay_buffer_capacity = 10000


    # Octo 模型路径 (按你本地下载位置修改)
    octo_path = os.environ.get("OCTO_PATH", "octo-small")

    def get_environment(self, fake_env=False, save_video=False,
                        classifier=False, stack_obs_num=1):
        env = LiftCubeSimEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        # 仿真中不接 SpacemouseIntervention
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                # 说明：这里的成功给10，其他给-0.05的奖励设计只对阶段二有效，阶段1中的demo奖励是成功给1，其他是0
                def sigmoid(x):
                    return 1 / (1 + jnp.exp(-x))
                # 分类器判定为成功 + 夹爪闭合(说明真的抓住了) + 末端抬高
                if (sigmoid(classifier_fn(obs)[0]) > 0.85
                        and env.unwrapped.curr_gripper_pos[0] < 0.0
                        and env.unwrapped.currpos[2] > 0.86):
                    return 10.0
                return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        env = GripperPenaltyWrapper(env, penalty=-0.2)
        return env
