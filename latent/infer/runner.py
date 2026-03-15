"""
推理编排器
协调 env（观测/动作）和 policy（策略推理），执行推理主循环
"""
import sys
import time

import numpy as np

from policies.base_policy import PolicyBase
from envs.robot_env import RobotEnvBase
from envs.link_robot_env import ActionSafetyError


class InferenceRunner:
    """推理编排器

    职责：
    - 从 env 获取观测
    - 调用 policy 预测动作序列
    - 将动作序列逐步下发给 env
    - 控制推理频率和步数
    """

    def __init__(self, env: RobotEnvBase, policy: PolicyBase,
                 max_steps: int = 1000, send_freq: float = 10.0):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.send_freq = send_freq

    def run(self) -> None:
        """执行推理主循环"""
        try:
            for step in range(self.max_steps):
                print(f"\n=== Step {step + 1}/{self.max_steps} ===")

                obs = self.env.get_obs()
                if not obs:
                    print("未获取到观测数据，跳过...")
                    continue

                action = self.policy.predict(obs)

                if action.ndim == 1:
                    action = action.reshape(1, -1)

                print(f"Action sequence: shape={action.shape}")

                for i, act in enumerate(action):
                    self.env.step(act)
                    print(f"  Step {i+1}/{len(action)}: action sent")
                    if self.send_freq > 0:
                        time.sleep(1.0 / self.send_freq)

                # 补偿运动延时
                time.sleep(0.21)

        except ActionSafetyError as e:
            print(f"\n[紧急停止] {e}")
            sys.exit(1)
