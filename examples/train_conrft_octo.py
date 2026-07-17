#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax.core import frozen_dict
import os
import copy
import pickle as pkl
import imageio
from PIL import Image

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from data_util import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

from serl_launcher.utils.launcher import (
    make_conrft_octo_cp_pixel_agent_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0,
                     "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")

flags.DEFINE_float("gamma", 0.95, "return discount")
flags.DEFINE_float("reward_neg", 0.0, "reward_neg for spase reward envs")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_float("q_weight", 0.1, "q_weight ")
flags.DEFINE_float("bc_weight", 1.0, "bc_weight")
# flags.DEFINE_float("trust_tau", 1.0, "trust temperature (越大越接近原始loss)")
# flags.DEFINE_float("trust_lambda", 1.0, "trust bc boost (不可信时BC抬升系数)")

flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(tasks, agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    '''
    这部分注释的代码是无法保存推理测试的视频，改为下面的代码可以保存视频
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        episode_length_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, _ = agent.sample_actions(
                    observations=jax.device_put(obs),
                    tasks=jax.device_put(tasks),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward > 0:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        episode_length_list.append(info["episode"]["l"])
                        print(dt)
                        print(info["episode"]["l"])

                    success_counter += 1 if reward > 0 else 0
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average episode length: {np.mean(episode_length_list)}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    '''
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        episode_length_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        video_dir = os.path.join(FLAGS.checkpoint_path, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            frames = []
            step_logs = []


            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, _ = agent.sample_actions(
                    observations=jax.device_put(obs),
                    tasks=jax.device_put(tasks),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                # ---- Q/V 推理 (train=False 不需要 dropout rng) ----
                obs_b = jax.tree_map(lambda x: x[None], jax.device_put(obs))  # 加 batch 维
                action_b = jnp.array(actions)[None]  # [1, 7]

                # Q 值: [ensemble_size, 1] -> 取 ensemble min -> scalar
                q_vals = agent.forward_critic(obs_b, None, action_b, rng=None, train=False)
                q_val = float(jax.device_get(q_vals).min())

                # V 值: [1, num_atoms] -> softmax -> 期望值
                v_logits = agent.forward_value(obs_b, rng=None, train=False)   # [1, num_atoms]
                v_probs = jax.nn.softmax(jax.device_get(v_logits), axis=-1)    # [1, num_atoms]
                atoms = np.linspace(
                    agent.config["v_min"], agent.config["v_max"], agent.config["num_atoms"])  # [num_atoms]

                # 算法实际用的: 自适应 tau 分位数 (需要先算熵)
                K = v_probs.shape[-1]
                ent = -np.sum(v_probs * np.log(v_probs + 1e-8), axis=-1)  # [1]
                norm_ent = ent / np.log(K)                                  # [1] in [0,1]
                tau = float(np.clip(
                    agent.config["tau_base"] - agent.config["tau_alpha"] * norm_ent, 0.01, 0.99))

                cdf = np.cumsum(v_probs[0])                        # [num_atoms]
                idx = int(np.argmax(cdf >= tau))
                v_quantile = float(atoms[idx])                     # ← 这才是算法用的 v_lower

                # 期望值(可选,用于对比)
                v_mean = float(np.sum(v_probs[0] * atoms))

                print(f"  step={len(step_logs):3d}  q={q_val:.4f}  v_lower={v_quantile:.4f}(τ={tau:.3f})  v_mean={v_mean:.4f}  ent={float(norm_ent):.3f}")
                step_logs.append({
                    "action": actions.tolist(),
                    "q": q_val,
                    "v_lower": v_quantile,
                    "v_mean": v_mean,
                    "tau": tau,
                    "norm_ent": float(norm_ent),
                })

                # 收集当前帧
                side = obs["side_policy_256"][-1]    # (256,256,3)
                wrist = obs["wrist_1"][-1]           # (128,128,3)
                wrist_resized = np.array(Image.fromarray(wrist).resize((256, 256), Image.BILINEAR))
                frame = np.concatenate([side, wrist_resized], axis=1)
                frames.append(frame)

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward > 0:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        episode_length_list.append(info["episode"]["l"])
                        print(dt)
                        print(info["episode"]["l"])

                    success_counter += 1 if reward > 0 else 0
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

            # 保存当前 episode 视频
            tag = "success" if reward > 0 else "fail"
            video_path = os.path.join(video_dir, f"ep{episode}_{tag}.mp4")
            imageio.mimwrite(video_path, frames, fps=20, quality=8)
            
            import json
            log_path = os.path.join(video_dir, f"ep{episode}_{tag}_qv.json")
            with open(log_path, "w") as f:
                json.dump(step_logs, f, indent=2)
            print(f"Q/V日志已保存: {log_path}")



            print(f"视频已保存: {video_path} ({len(frames)} 帧)")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average episode length: {np.mean(episode_length_list)}")
        print(f"average time: {np.mean(time_list)}")
        print(f"所有评估视频保存在: {video_dir}")
        return  # after done eval, return and exit

    start_step = (
        #int(os.path.basename(natsorted(glob.glob(os.path.join( # 无课程学习的读取方式
        #    FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        int(os.path.basename(natsorted(glob.glob(os.path.join(  # 课程学习递归读取并取出最后一个
            FLAGS.checkpoint_path, "buffer", "**", "*.pkl"), recursive=True))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    #obs, _ = env.reset()              # 不使用逆向课程学习
    #obs, _ = env.reset(options={"current_step": 0}) # 使用逆向课程学习，但每次启动都从0开始
    obs, _ = env.reset(options={"current_step": start_step}) # 使用逆向课程学习
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    trajectory = []

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, action_embeddings = agent.sample_actions(
                    observations=jax.device_put(obs),
                    tasks=jax.device_put(tasks),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                intervened=already_intervened,
                embeddings=action_embeddings,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']

            trajectory.append(transition)

            obs = next_obs
            if done or truncated:
                trajectory = add_mc_returns_to_trajectory(trajectory, FLAGS.gamma,
                                                          FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.reward_neg, is_sparse_reward=False
                                                          )
                trajectory = add_next_embeddings_to_trajectory(trajectory)
                for transition in trajectory:
                    data_store.insert(transition)
                    # 只有在启用 buffer 保存时，才在内存中累积 transitions，防止 buffer_period=0 时内存泄漏
                    if config.buffer_period > 0:
                        transitions.append(copy.deepcopy(transition))
                        if transition['intervened']:
                            demo_transitions.append(copy.deepcopy(transition))
                    
                    if transition['intervened']:
                        intvn_data_store.insert(transition)

                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["succeed"] = int(info['succeed'])
                info["episode"]["total_steps"] = step
                # send stats to the learner to log
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                trajectory = []
                # obs, _ = env.reset()                # 不使用逆向课程学习
                # obs, _ = env.reset(options={"current_step": step - start_step}) # 使用逆向课程学习
                obs, _ = env.reset(options={"current_step": step}) # 使用逆向课程学习

        '''
        # ------------------------------------------ #
        # 无课程学习的buffer保留
        # ------------------------------------------ #
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(
                FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []
        '''

        # ------------------------------------------ #
        # 课程学习的buffer保留，每个课程保留几个buffer
        # ------------------------------------------ #
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            MAX_FILES_PER_STAGE = 3   # 每个课程阶段最多保留几个 buffer 文件

            # 用相对步数判断当前处于哪个课程阶段（与 reset 里的阈值保持一致）
            #rel_step = step - start_step # 使用这句每次启动都从0从头开始训练
            rel_step = step
            if rel_step < 10000:
                stage = "stage1_lift"        # 第一关：已夹住，只学抬起
            elif rel_step < 20000:
                stage = "stage2_close_lift"  # 第二关：套在外侧，学合拢+抬起
            else:
                stage = "stage3_full"        # 第三关：标准完整任务

            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer", stage)
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer", stage)
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)

            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

            # 每个阶段只保留最近 MAX_FILES_PER_STAGE 个文件，删掉更早的
            for folder in (buffer_path, demo_buffer_path):
                files = sorted(
                    glob.glob(os.path.join(folder, "transitions_*.pkl")),
                    key=lambda p: int(os.path.basename(p)[12:-4]),
                )
                for old_file in files[:-MAX_FILES_PER_STAGE]:
                    os.remove(old_file)

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, tasks, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # ------------------------------------------------------------------------------------------- #
    # 首先读取checkpoint的步数，从而判断进行阶段几的训练：大于一个阈值直接进行阶段2，否则则进行阶段1.
    # ------------------------------------------------------------------------------------------- #
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(
            FLAGS.checkpoint_path))[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step
    online_start_step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    train_critic_networks_to_update = frozenset({"critic", "value"})
    train_actor_networks_to_update = frozenset({"actor"})
    train_networks_to_update = frozenset({"critic", "actor", "value"})

    def create_batch_tasks(data_dict, batch_size):
        batch_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):  # Handling nested dictionary (e.g., language_instruction)
                batch_dict[key] = {k: np.tile(
                    v, (batch_size, *([1] * (v.ndim - 1)))) for k, v in value.items()}
            else:
                # For non-dictionary values, repeat along batch dimension (axis=0)
                batch_dict[key] = np.tile(
                    value, (batch_size, *([1] * (value.ndim - 1))))  # Repeat along axis 0

        return batch_dict

    # Pretrain the model with the demo data
    if step < FLAGS.pretrain_steps:
        print_green("Pretraining the model with demo data")

        # ---- 本地记录自适应系数 + loss,训练结束后画图 ----
        metric_history = {
            "step": [],
            "actor_loss": [], "q_loss": [], "bc_loss": [], "mse": [],
            "value_loss": [], "value_expected_mean": [], "value_entropy": [],
            "value_bound_tau": [], "value_lower_bound": [], "calql_bound_rate": [],
        }

        for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), desc="pretraining"):
            for _ in range(config.cta_ratio - 1):
                batch = next(demo_buffer.get_iterator(
                    sample_args={"batch_size": config.batch_size,
                                 "pack_obs": True, },
                    device=sharding.replicate(),
                ))

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
                batch = frozen_dict.freeze(batch)
                agent, critics_info = agent.update_calql(
                    batch, networks_to_update=train_critic_networks_to_update,)

            batch = next(demo_buffer.get_iterator(
                sample_args={"batch_size": config.batch_size,
                             "pack_obs": True, },
                device=sharding.replicate(),
            ))

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)

            agent, update_info = agent.update_calql(
                batch, networks_to_update=train_networks_to_update,)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)

            if step % config.log_period == 0:
                actor_info = update_info.get("actor", update_info)
                critic_info = update_info.get("critic", {})
                value_info = update_info.get("value", {})
                metric_history["step"].append(int(step))
                for k in ("actor_loss", "q_loss", "bc_loss", "mse"):
                    metric_history[k].append(float(actor_info[k]))
                # value 网络指标
                for k in ("value_loss", "value_expected_mean", "value_entropy"):
                    metric_history[k].append(float(value_info.get(k, 0.0)))
                # 自适应下界/tau 指标 (在 critic info 中)
                for k in ("value_bound_tau", "value_lower_bound", "calql_bound_rate"):
                    metric_history[k].append(float(critic_info.get(k, 0.0)))

            if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=step, keep=100)

        
        # ---- 训练结束:本地绘制曲线(自适应系数包络 + std + loss) ----
        try:
            import csv
            import matplotlib
            matplotlib.use("Agg")          # 无显示环境也能出图
            import matplotlib.pyplot as plt

            out_dir = FLAGS.checkpoint_path
            os.makedirs(out_dir, exist_ok=True)
            steps = metric_history["step"]

            def _plot_envelope(ax, prefix, title):
                mean = metric_history[f"{prefix}_mean"]
                lo   = metric_history[f"{prefix}_min"]
                hi   = metric_history[f"{prefix}_max"]
                ax.plot(steps, mean, label="mean", color="C0")
                ax.fill_between(steps, lo, hi, alpha=0.2, color="C0", label="min~max")
                ax.set_title(title)
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)


            # 图3: loss 曲线 bc_loss / q_loss / actor_loss
            fig3, axes3 = plt.subplots(1, 5, figsize=(25, 4))
            axes3[0].plot(steps, metric_history["bc_loss"], color="C0")
            axes3[0].set_title("bc_loss (consistency, 高方差)")
            axes3[1].plot(steps, metric_history["q_loss"], color="C1")
            axes3[1].set_title("q_value (mean Q)")
            axes3[2].plot(steps, metric_history["actor_loss"], color="C2")
            axes3[2].set_title("actor_loss (total)")
            axes3[3].plot(steps, metric_history["mse"], color="C3")
            axes3[3].set_title("mse (动作预测误差, 低方差, 看这条)")
            axes3[4].plot(steps, metric_history["value_loss"], color="C4")
            axes3[4].set_title("value_loss (DIVL, C51 交叉熵)")
            for ax in axes3:
                ax.set_xlabel("step"); 
                ax.grid(True, alpha=0.3)
            fig3.tight_layout()
            loss_png_path = os.path.join(out_dir, "loss_curve.png")
            fig3.savefig(loss_png_path, dpi=150)
            plt.close(fig3)

            # 图4: DIVL value 网络 & 自适应 tau 曲线
            fig4, axes4 = plt.subplots(2, 3, figsize=(18, 8))
            axes4[0, 0].plot(steps, metric_history["value_loss"], color="C0")
            axes4[0, 0].set_title("value_loss (C51 交叉熵)")
            axes4[0, 1].plot(steps, metric_history["value_expected_mean"], color="C1")
            axes4[0, 1].set_title("value_expected_mean (V 分布期望)")
            axes4[0, 2].plot(steps, metric_history["value_entropy"], color="C2")
            axes4[0, 2].set_title("value_entropy (归一化熵, 越大越不确定)")
            axes4[1, 0].plot(steps, metric_history["value_bound_tau"], color="C3")
            axes4[1, 0].set_title("value_bound_tau (自适应分位数 tau)")
            axes4[1, 1].plot(steps, metric_history["value_lower_bound"], color="C4")
            axes4[1, 1].set_title("value_lower_bound (Cal-QL 下界)")
            axes4[1, 2].plot(steps, metric_history["calql_bound_rate"], color="C5")
            axes4[1, 2].set_title("calql_bound_rate (被下界夹住比例)")
            for ax in axes4.flat:
                ax.set_xlabel("step")
                ax.grid(True, alpha=0.3)
            fig4.tight_layout()
            value_png_path = os.path.join(out_dir, "value_tau_curve.png")
            fig4.savefig(value_png_path, dpi=150)
            plt.close(fig4)
            print_green(f"[绘图] value/tau曲线图: {value_png_path}")

            # 原始数据存 CSV
            csv_path = os.path.join(out_dir, "adaptive_weights_curve.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(list(metric_history.keys()))
                for row in zip(*metric_history.values()):
                    w.writerow(row)

            print_green(f"[绘图] loss曲线图: {loss_png_path}")
            print_green(f"[绘图] 原始数据CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] 绘制曲线失败: {e}")
        print_green("Pretraining done")
        return  # after pretraining, return and exit
    else:
        print_green(
            "Existing pretrained checkpoint model found. Skipping pretraining")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # Start online training after offline pretraining
    online_start_step = FLAGS.pretrain_steps + \
        1 if online_start_step < FLAGS.pretrain_steps else online_start_step
    for step in tqdm.tqdm(range(online_start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
            batch = frozen_dict.freeze(batch)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_ql(
                    batch, networks_to_update=train_critic_networks_to_update,)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update_ql(
                batch, networks_to_update=train_networks_to_update,)
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=step, keep=100)


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner, save_video=FLAGS.eval_checkpoint_step, classifier=(FLAGS.pretrain_steps == 0), stack_obs_num=1)
    env = RecordEpisodeStatistics(env)

    FLAGS.reward_neg = config.reward_neg

    rng, sampling_rng = jax.random.split(rng)

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    if config.setup_mode == 'single-arm-fixed-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            fix_gripper=True,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = False
        include_octo_embeddings = True
        include_mc_returns = True
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = True
        include_octo_embeddings = True
        include_mc_returns = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(
        jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        if not FLAGS.learner:
            input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state,)
        # agent = agent.replace(state=ckpt)

        # Update params only, ignore the optimizer states
        new_params = ckpt.params
        new_target_params = ckpt.target_params

        agent = agent.replace(state=agent.state.replace(
            params=new_params, target_params=new_target_params))

        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(FLAGS.checkpoint_path))[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        # set up wandb and logging

        wandb_logger = make_wandb_logger(
            project="conrft",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        assert FLAGS.demo_path is not None

        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    if include_grasp_penalty and 'grasp_penalty' not in transition:
                        transition['grasp_penalty'] = 0.0
                    if include_octo_embeddings:
                        if 'embeddings' not in transition:
                            transition['embeddings'] = np.zeros(384, dtype=np.float32)
                        if 'next_embeddings' not in transition:
                            transition['next_embeddings'] = np.zeros(384, dtype=np.float32)
                    if include_mc_returns and 'mc_returns' not in transition:
                        transition['mc_returns'] = 0.0
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            #for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer", "**", "*.pkl"), recursive=True):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            #for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer", "**", "*.pkl"), recursive=True):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng,
                tasks,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
                )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(tasks,
              agent,
              data_store,
              intvn_data_store,
              env,
              sampling_rng,
              )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
