"""
VITA 推理主类
"""
from collections import deque
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from thor.engine.loader import load_vita_policy
from thor.engine.runner import VitaRunner


class VitaInference:
    """VITA 推理主类"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        mixed_precision: str = "bf16",
        device: str = "cuda",
    ):
        """
        初始化 VITA 推理
        
        Args:
            checkpoint_dir: checkpoint 目录路径
            mixed_precision: 混合精度类型 (bf16, fp16, fp32)
            device: 运行设备
        """
        if mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.device = device
        self.policy = load_vita_policy(checkpoint_dir, device=device)
        self.policy.eval()
        
        # 获取模型配置
        obs_horizon = getattr(self.policy.config.policy, 'obs_horizon', 1)
        action_horizon = getattr(self.policy.config.policy, 'action_horizon', 8)
        
        self.runner = VitaRunner(
            obs_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        
        self.action_horizon = action_horizon
        self._action_queue = deque(maxlen=action_horizon)
    
    def update_obs(self, observation: Dict[str, Any]):
        """更新观测"""
        self.runner.update_obs(observation)
    
    def predict_action(self, observation: Optional[Dict[str, Any]] = None):
        """
        预测动作
        
        Args:
            observation: 当前观测（可选）
        
        Returns:
            动作序列
        """
        if self.dtype == torch.float32:
            return self.runner.get_action(self.policy, observation)
        
        with torch.autocast(device_type=self.device.split(':')[0], dtype=self.dtype):
            return self.runner.get_action(self.policy, observation)
    
    def get_action(self, observation: Optional[Dict[str, Any]] = None):
        """获取动作（predict_action 的别名）"""
        return self.predict_action(observation)
    
    def reset(self):
        """重置推理状态"""
        self.runner.reset_obs()
        self.policy.reset()
        self._action_queue.clear()
    
    def get_last_obs(self):
        """获取最后一次观测"""
        if len(self.runner.obs) > 0:
            return self.runner.obs[-1]
        return None
    
    @classmethod
    def from_config(cls, cfg):
        """从配置创建实例"""
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        return cls(
            checkpoint_dir=cfg.checkpoint_dir,
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            device=cfg.get("device", "cuda"),
        )


__all__ = ["VitaInference"]

