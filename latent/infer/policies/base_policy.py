"""策略抽象基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class PolicyBase(ABC):
    """策略抽象基类

    子类需要实现：
    - predict(obs): 接收观测字典，返回动作序列
    - reset(): 重置策略状态
    """

    @abstractmethod
    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """预测动作序列

        Args:
            obs: 观测字典，格式为 {"observation": {"images": {...}, "state": ndarray}}

        Returns:
            动作序列 (action_horizon, action_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """重置策略内部状态"""
        raise NotImplementedError
