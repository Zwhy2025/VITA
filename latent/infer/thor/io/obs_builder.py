"""
观测构建器
提供统一的观测数据构建功能
"""
from typing import Any, Dict, Optional

import numpy as np

from thor.utils.image import process_image
from thor.utils.config import merge_nested_dict


def get_image_config(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """从映射配置中提取图像配置"""
    image_cfg = mapping.get("image", {})
    return {
        "expected_size": tuple(image_cfg.get("expected_size", None)) if image_cfg.get("expected_size") else None,
        "channel_order": image_cfg.get("channel_order", "rgb"),
        "normalize": bool(image_cfg.get("normalize", True)),
        "resize": bool(image_cfg.get("resize", False)),
    }


def process_image_obs(
    rgb: Optional[np.ndarray],
    model_key: str,
    image_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """处理图像观测并转换为嵌套结构
    
    Args:
        rgb: RGB 图像数组
        model_key: 模型键名（如 "observation.images.front"）
        image_cfg: 图像配置
    
    Returns:
        嵌套的观测字典
    """
    chw = process_image(
        rgb=rgb,
        expected_size=image_cfg["expected_size"],
        channel_order=image_cfg["channel_order"],
        normalize=image_cfg["normalize"],
        resize=image_cfg["resize"],
    )
    
    # 将点分隔键转换为嵌套结构
    nested = {}
    parts = model_key.split(".")
    current = nested
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = chw
    
    return nested


def build_state_obs(
    qpos: np.ndarray,
    model_arm_dof: int,
    use_gripper: bool,
) -> np.ndarray:
    """构建状态观测（简单模式，直接使用 qpos）
    
    Args:
        qpos: 关节位置数组
        model_arm_dof: 模型期望的手臂自由度
        use_gripper: 是否包含夹爪
    
    Returns:
        处理后的状态数组
    """
    expected_dim = model_arm_dof + (1 if use_gripper else 0)
    
    if len(qpos) < expected_dim:
        # 填充零
        qpos_padded = np.zeros(expected_dim, dtype=np.float32)
        qpos_padded[:len(qpos)] = qpos[:expected_dim]
        return qpos_padded
    elif len(qpos) > expected_dim:
        # 截断
        return qpos[:expected_dim].astype(np.float32)
    else:
        return qpos.astype(np.float32)


__all__ = [
    "get_image_config",
    "process_image_obs",
    "build_state_obs",
]

