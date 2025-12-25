"""
VITA 模型推理模块
基于 flare 包中的 VitaPolicy 进行推理
"""
import os
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file


def dict_apply(x: Dict[str, Any], func):
    """递归应用函数到字典中的所有值"""
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def load_vita_policy(checkpoint_dir: str, device: str = "cuda", config_yaml_path: Optional[str] = None):
    """
    从 checkpoint 目录加载 VITA 模型
    
    Args:
        checkpoint_dir: checkpoint 目录路径，包含 model.safetensors
        device: 运行设备
        config_yaml_path: 训练配置 YAML 文件路径（可选，默认从 logs 目录查找）
    
    Returns:
        加载好的 VitaPolicy 模型
    """
    from flare.policies.vita.vita_policy import VitaPolicy
    from flare.factory import get_policy_class
    import yaml
    
    # 查找训练配置 YAML 文件
    if config_yaml_path is None:
        # 尝试从 checkpoint 目录的父目录的 logs 目录查找
        checkpoint_path = Path(checkpoint_dir)
        logs_dir = checkpoint_path.parent / "logs"
        
        # 查找最新的 train_config_*.yaml
        config_files = list(logs_dir.glob("train_config_*.yaml"))
        if config_files:
            config_yaml_path = str(max(config_files, key=lambda p: p.stat().st_mtime))
        else:
            raise FileNotFoundError(
                f"Config YAML not found. Please specify config_yaml_path or ensure "
                f"train_config_*.yaml exists in {logs_dir}"
            )
    
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Config YAML file not found: {config_yaml_path}")
    
    # 加载训练配置
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # 转换为 OmegaConf
    config = OmegaConf.create(config_dict)
    
    # 从配置中提取 stats（归一化统计量）
    # 优先使用 override_stats，如果没有则从数据集加载
    stats = {}
    
    # 首先尝试从原始数据集加载基础 stats（从 meta/stats.json）
    try:
        # 从 repo_id 构建原始数据集路径
        repo_id = config.task.dataset_repo_id
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join("/root/workspace/VITA/datasets", repo_id),
            os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), "datasets", repo_id),
            os.path.expandvars(config.task.dataset_root),
        ]
        
        stats_json_path = None
        for base_path in possible_paths:
            test_path = os.path.join(base_path, "meta", "stats.json")
            if os.path.exists(test_path):
                stats_json_path = test_path
                break
        
        if stats_json_path:
            with open(stats_json_path, "r") as f:
                dataset_stats = json.load(f)
            
            # 数据集 stats 可能包含 min/max，需要转换为 mean/std
            for key, stat_dict in dataset_stats.items():
                if "mean" in stat_dict and "std" in stat_dict:
                    # 已经有 mean/std，直接使用
                    stats[key] = {
                        "mean": torch.tensor(stat_dict["mean"], dtype=torch.float32),
                        "std": torch.tensor(stat_dict["std"], dtype=torch.float32),
                    }
                elif "min" in stat_dict and "max" in stat_dict:
                    # 从 min/max 计算 mean/std
                    min_vals = np.array(stat_dict["min"], dtype=np.float32)
                    max_vals = np.array(stat_dict["max"], dtype=np.float32)
                    mean_vals = (min_vals + max_vals) / 2.0
                    # std 估算：使用范围的一半作为 std（假设均匀分布）
                    std_vals = (max_vals - min_vals) / 4.0
                    # 避免 std 为 0
                    std_vals = np.maximum(std_vals, 1e-6)
                    stats[key] = {
                        "mean": torch.from_numpy(mean_vals),
                        "std": torch.from_numpy(std_vals),
                    }
        else:
            # 如果找不到原始数据集，尝试使用 AVAlohaDatasetMeta
            try:
                from gym_av_aloha.datasets.av_aloha_dataset import AVAlohaDatasetMeta
                dataset_meta = AVAlohaDatasetMeta(
                    repo_id=config.task.dataset_repo_id,
                    root=os.path.expandvars(config.task.dataset_root)
                )
                dataset_stats = dataset_meta.stats
                
                # 数据集 stats 可能包含 min/max，需要转换为 mean/std
                for key, stat_dict in dataset_stats.items():
                    if "mean" in stat_dict and "std" in stat_dict:
                        stats[key] = {
                            "mean": torch.tensor(stat_dict["mean"], dtype=torch.float32),
                            "std": torch.tensor(stat_dict["std"], dtype=torch.float32),
                        }
                    elif "min" in stat_dict and "max" in stat_dict:
                        min_vals = np.array(stat_dict["min"], dtype=np.float32)
                        max_vals = np.array(stat_dict["max"], dtype=np.float32)
                        mean_vals = (min_vals + max_vals) / 2.0
                        std_vals = (max_vals - min_vals) / 4.0
                        std_vals = np.maximum(std_vals, 1e-6)
                        stats[key] = {
                            "mean": torch.from_numpy(mean_vals),
                            "std": torch.from_numpy(std_vals),
                        }
            except Exception as e2:
                print(f"Warning: Failed to load stats via AVAlohaDatasetMeta: {e2}")
    except Exception as e:
        # 如果数据集不可访问，继续使用 override_stats
        print(f"Warning: Failed to load stats from dataset: {e}")
    
    # 使用 override_stats 覆盖（如果存在）
    if hasattr(config.task, 'override_stats') and config.task.override_stats:
        for key, stat_dict in config.task.override_stats.items():
            # 将嵌套列表转换为 numpy array 再转为 tensor
            mean_array = np.array(stat_dict["mean"], dtype=np.float32)
            std_array = np.array(stat_dict["std"], dtype=np.float32)
            stats[key] = {
                "mean": torch.from_numpy(mean_array),
                "std": torch.from_numpy(std_array),
            }
    
    # 确保所有需要的键都有统计信息
    # 检查 image_keys 和 state_key
    required_keys = list(getattr(config.task, 'image_keys', [])) + [config.task.state_key]
    missing_keys = [key for key in required_keys if key not in stats]
    
    if missing_keys:
        raise RuntimeError(
            f"Missing stats for required keys: {missing_keys}. "
            f"Please ensure override_stats includes these keys or dataset is accessible."
        )
    
    # 如果仍然没有 stats，报错
    if not stats:
        raise RuntimeError(
            "No stats available. Please ensure override_stats is set in config "
            "or dataset is accessible."
        )
    
    # 创建模型
    policy_cls = get_policy_class(config.policy.name)
    policy = policy_cls(config, stats)
    
    # 加载权重
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        policy.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    policy.to(device)
    policy.eval()
    policy.reset()
    
    return policy


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


__all__ = ["VitaInference", "VitaRunner", "load_vita_policy"]

