#!/usr/bin/env python3
"""将 demo pickle 数据可视化为视频"""
import pickle
import numpy as np
import imageio

def main():
    demo_path = "/home/robot/sjl/conrft-main/examples/demo_data/task1_lift_cube_sim_30_demos.pkl"
    output_path = "/home/robot/sjl/conrft-main/examples/demo_data/demo_visualization.mp4"

    with open(demo_path, "rb") as f:
        transitions = pickle.load(f)

    print(f"共 {len(transitions)} 个 transitions")

    frames = []
    for i, t in enumerate(transitions):
        side = t["observations"]["side_policy_256"][-1]    # (256,256,3) uint8 RGB
        wrist = t["observations"]["wrist_1"][-1]           # (128,128,3) uint8 RGB

        # resize wrist 到 256x256
        from PIL import Image
        wrist_pil = Image.fromarray(wrist).resize((256, 256), Image.BILINEAR)
        wrist_resized = np.array(wrist_pil)

        # 水平拼接
        frame = np.concatenate([side, wrist_resized], axis=1)  # (256, 512, 3)
        frames.append(frame)

    imageio.mimwrite(output_path, frames, fps=20, quality=8)
    print(f"视频已保存: {output_path} ({len(frames)} 帧)")

if __name__ == "__main__":
    main()