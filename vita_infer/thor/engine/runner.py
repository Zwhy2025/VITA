"""
VITA 推理运行器
管理观测历史和动作生成
"""
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from thor.engine.loader import dict_apply


class VitaRunner:
    """VITA 推理运行器，管理观测历史和动作队列"""
    
    def __init__(self, obs_horizon: int = 1, action_horizon: int = 8):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs = deque(maxlen=obs_horizon + 1)
    
    def reset_obs(self):
        """重置观测历史"""
        self.obs.clear()
    
    def update_obs(self, current_obs: Dict[str, Any]):
        """更新观测"""
        self.obs.append(current_obs)
    
    def _stack_last_n_obs(self, all_obs, n_steps):
        """堆叠最后 n 步观测"""
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result
    
    def _get_n_steps_obs(self):
        """获取 n 步观测"""
        assert len(self.obs) > 0, "No observation recorded, please update obs first"
        
        def _recursive_stack(obs_list, n_steps):
            if not obs_list:
                return {}
            
            result = {}
            first_obs = obs_list[0]
            if isinstance(first_obs, dict):
                for key in first_obs.keys():
                    values = [obs[key] if isinstance(obs, dict) and key in obs else None for obs in obs_list]
                    values = [v for v in values if v is not None]
                    if values:
                        if isinstance(values[0], dict):
                            result[key] = _recursive_stack(values, n_steps)
                        else:
                            result[key] = self._stack_last_n_obs(values, n_steps)
            return result
        
        return _recursive_stack(self.obs, self.obs_horizon)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """将嵌套字典展平为点分隔键格式"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_action(self, policy, observation: Optional[Dict[str, Any]] = None):
        """
        获取动作
        
        Args:
            policy: VitaPolicy 模型
            observation: 当前观测（可选，如果为 None 则使用已缓存的观测）
        
        Returns:
            动作序列 (action_horizon, action_dim)
        """
        device = next(policy.parameters()).device
        
        if observation is not None:
            self.obs.append(observation)
        
        obs = self._get_n_steps_obs()
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        
        # 展平为点分隔键格式
        obs_dict_flat = self._flatten_dict(obs_dict)
        
        # VitaPolicy.generate_actions 期望 [b, ...] 格式（没有 sequence 维度）
        # 然后我们会添加 sequence 维度：v.unsqueeze(1) -> [b, 1, ...]
        # BaseObserver.get_images 期望 [b, s, n, c, h, w]，其中 n 是图像数量
        # 所以对于图像键，应该是 [b, c, h, w]
        # 添加 sequence 维度后会变成 [b, 1, c, h, w]
        # 然后 get_images 会 stack 多个图像键，变成 [b, 1, n, c, h, w]
        obs_dict_input = {}
        for k, v in obs_dict_flat.items():
            # _get_n_steps_obs 返回的是 [s, ...] 格式（s = obs_horizon）
            # 我们需要取最后一个 sequence，得到 [...] 格式，然后添加 batch 维度
            if v.ndim >= 1:
                if v.shape[0] == self.obs_horizon:
                    # [s, ...] -> [...] -> [1, ...] (取最后一个 sequence，添加 batch 维度)
                    obs_dict_input[k] = v[-1].unsqueeze(0)
                elif v.ndim >= 2 and v.shape[0] == 1 and v.shape[1] == self.obs_horizon:
                    # [1, s, ...] -> [1, ...] (取最后一个 sequence)
                    obs_dict_input[k] = v[:, -1]
                elif v.ndim >= 1 and v.shape[0] == 1:
                    # 已经是 [1, ...] 格式，保持不变
                    obs_dict_input[k] = v
                else:
                    # [...] -> [1, ...] (添加 batch 维度)
                    obs_dict_input[k] = v.unsqueeze(0)
            else:
                obs_dict_input[k] = v
        
        with torch.no_grad():
            # 直接调用 generate_actions 获取完整的动作序列
            # 需要先添加 sequence 维度，然后归一化
            batch = {k: v.unsqueeze(1) for k, v in obs_dict_input.items() 
                    if k in policy.config.task.image_keys + [policy.config.task.state_key]}
            batch = policy.normalize_inputs(batch)
            
            # 生成动作序列 (batch_size, pred_horizon, action_dim)
            actions = policy.generate_actions(batch)
            
            # 取前 action_horizon 个动作
            actions = actions[:, :self.action_horizon]
            
            # 反归一化
            actions = policy.unnormalize_outputs({"action": actions})["action"]
        
        # 转换为 numpy: (batch_size, action_horizon, action_dim) -> (action_horizon, action_dim)
        actions = actions.detach().cpu().float().numpy()
        if actions.shape[0] == 1:
            actions = actions[0]  # 去掉 batch 维度
        
        return actions


__all__ = ["VitaRunner"]

