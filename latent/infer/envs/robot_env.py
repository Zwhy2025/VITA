"""
机器人环境抽象基类
定义推理循环所需的统一接口，具体实现可适配不同机器人 SDK
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class RobotEnvBase(ABC):
    """机器人环境抽象基类

    子类需要实现：
    - get_obs(): 返回模型所需的观测字典
    - step(action): 执行一步动作
    - start() / stop(): 生命周期管理
    """

    @abstractmethod
    def get_obs(self) -> Dict[str, Any]:
        """获取当前观测

        Returns:
            模型期望的观测字典，格式为:
            {
                "observation": {
                    "images": {
                        "front": ndarray (C, H, W),
                        ...
                    },
                    "state": ndarray (state_dim,)
                }
            }
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """执行一步动作

        Args:
            action: 动作数组 (action_dim,)
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """启动环境（连接硬件等）"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """停止环境（释放硬件等）"""
        raise NotImplementedError
