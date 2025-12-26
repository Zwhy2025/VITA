"""
配置工具函数
"""
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(global_cfg: Dict[str, Any], scene_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并全局配置和场景配置
    
    Args:
        global_cfg: 全局配置字典
        scene_cfg: 场景配置字典（可选）
    
    Returns:
        合并后的配置字典
    """
    if scene_cfg is None:
        return global_cfg.copy()
    
    merged = global_cfg.copy()
    # 深度合并场景配置
    for key, value in scene_cfg.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def build_nested_dict(key: str, value: Any) -> Dict[str, Any]:
    """将点分隔的键转换为嵌套字典
    
    例如: "observation.images.image" -> {"observation": {"images": {"image": value}}}
    """
    parts = key.split(".")
    result = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def merge_nested_dict(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """合并嵌套字典"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            merge_nested_dict(target[key], value)
        else:
            target[key] = value
    return target


__all__ = ["load_config", "merge_configs", "build_nested_dict", "merge_nested_dict"]

