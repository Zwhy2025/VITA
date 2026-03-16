"""
VITA 策略实现
负责：加载模型、输入预处理、调用 flow matching 生成动作、反归一化
"""
from contextlib import nullcontext
from typing import Any, Dict

import numpy as np
import torch

from policies.base_policy import PolicyBase
from policies.loader import load_vita_policy, dict_apply


class VitaPolicy(PolicyBase):
    """VITA flow matching 策略

    职责边界：
    - 加载训练好的模型权重
    - 将 numpy obs dict 转为 torch tensor
    - normalize → generate_actions → unnormalize
    - 返回 numpy 动作序列

    不负责：obs 历史管理、推理循环编排（由 InferenceRunner 负责）
    """

    def __init__(self, checkpoint_dir: str, mixed_precision: str = "bf16", device: str = "cuda"):
        self._device = device

        if mixed_precision == "bf16":
            self._dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        self._model = load_vita_policy(checkpoint_dir, device=device)
        self.action_horizon = getattr(self._model.config.policy, 'action_horizon', 8)

    def predict(self, obs: Dict[str, Any]) -> np.ndarray:
        """接收观测字典，返回动作序列

        Args:
            obs: 嵌套字典，如 {"observation": {"images": {"front": ndarray}, "state": ndarray}}

        Returns:
            动作序列 (action_horizon, action_dim)
        """
        device = next(self._model.parameters()).device
        device_type = device.type

        # numpy → torch
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        obs_flat = _flatten_dict(obs_dict)

        # 添加 batch 维度: [...] → [1, ...]
        obs_batch = {}
        for k, v in obs_flat.items():
            if v.ndim >= 1 and v.shape[0] != 1:
                obs_batch[k] = v.unsqueeze(0)
            else:
                obs_batch[k] = v

        if self._dtype != torch.float32:
            ctx = torch.autocast(device_type=device_type, dtype=self._dtype)
        else:
            ctx = nullcontext()
        with torch.no_grad(), ctx:
            # 添加 sequence 维度，筛选模型需要的 key
            valid_keys = self._model.config.task.image_keys + [self._model.config.task.state_key]
            batch = {k: v.unsqueeze(1) for k, v in obs_batch.items() if k in valid_keys}
            batch = self._model.normalize_inputs(batch)
            actions = self._model.generate_actions(batch)
            actions = actions[:, :self.action_horizon]
            actions = self._model.unnormalize_outputs({"action": actions})["action"]

        actions = actions.detach().cpu().float().numpy()
        if actions.shape[0] == 1:
            actions = actions[0]
        return actions

    def reset(self) -> None:
        self._model.reset()


def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """将嵌套字典展平为点分隔键"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
