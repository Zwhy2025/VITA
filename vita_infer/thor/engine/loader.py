"""
VITA 模型加载模块
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
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


__all__ = ["load_vita_policy", "dict_apply"]

