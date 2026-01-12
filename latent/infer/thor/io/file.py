"""
文件数据源
从本地文件读取观测数据
"""
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from thor.io.obs_builder import get_image_config, process_image_obs, build_state_obs
from thor.utils.config import merge_nested_dict


def load_image(image_path: Path) -> np.ndarray:
    """加载图像文件"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_qpos(qpos_path: Path) -> np.ndarray:
    """从 CSV 文件加载 qpos"""
    with open(qpos_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        row = next(reader)  # 读取第一行数据
        # 解析数据，去除空格
        qpos = [float(x.strip()) for x in row]
    return np.array(qpos, dtype=np.float32)


def find_image_files(offline_dir: Path, image_name: str) -> List[Path]:
    """查找匹配的图像文件
    
    Args:
        offline_dir: 离线数据目录
        image_name: 图像名称（如 "front", "wrist_right"）
    
    Returns:
        匹配的图像文件列表
    """
    # 根据配置映射，查找对应的图像文件
    # 支持多种命名方式：
    # - 文件名包含 image_name
    # - 文件名包含相机名称的变体
    name_mapping = {
        "front": ["front", "color", "color1"],
        "wrist_right": ["wrist_right", "wrist", "wrist_camera"],
        "wrist_left": ["wrist_left", "wrist"],
        "head": ["head", "color", "color1"],
    }
    
    search_names = name_mapping.get(image_name, [image_name])
    
    # 支持的文件扩展名
    extensions = [".png", ".jpg", ".jpeg"]
    
    files = []
    # 先获取所有图像文件
    all_files = []
    for ext in extensions:
        all_files.extend(offline_dir.glob(f"*{ext}"))
        all_files.extend(offline_dir.glob(f"*{ext.upper()}"))
    
    # 然后根据名称过滤
    for name in search_names:
        for file_path in all_files:
            file_name_lower = file_path.name.lower()
            name_lower = name.lower()
            # 检查文件名是否包含搜索名称
            if name_lower in file_name_lower:
                files.append(file_path)
    
    # 去重并排序
    files = sorted(set(files))
    return files


def build_state_from_qpos(
    qpos: np.ndarray,
    model_arm_order: List[str],
    model_arm_dof: int,
    use_gripper: bool,
    fill_missing_arm: bool,
) -> np.ndarray:
    """从 qpos 构建模型期望的状态向量（支持多臂）
    
    Args:
        qpos: 关节位置数组，按 model_arm_order 顺序排列
        model_arm_order: 模型期望的机械臂顺序
        model_arm_dof: 模型期望的每个机械臂自由度（不含夹爪）
        use_gripper: 是否包含夹爪
        fill_missing_arm: 如果缺少某个机械臂数据，是否用零填充
    
    Returns:
        处理后的状态数组，按 model_arm_order 顺序拼接
    """
    stride = model_arm_dof + (1 if use_gripper else 0)
    arm_seg_len = stride
    
    # 按顺序分割 qpos
    segments: Dict[str, np.ndarray] = {}
    offset = 0
    for arm_name in model_arm_order:
        if offset + arm_seg_len <= len(qpos):
            segments[arm_name] = qpos[offset : offset + arm_seg_len]
        elif offset < len(qpos):
            # 如果数据不足，取剩余部分
            segments[arm_name] = qpos[offset:]
        else:
            segments[arm_name] = None
        offset += arm_seg_len
    
    # 构建输出数组
    out = []
    for name in model_arm_order:
        seg = segments.get(name)
        if seg is None:
            if fill_missing_arm:
                seg = np.zeros(stride, dtype=np.float32)
            else:
                raise RuntimeError(f"missing arm in observation: {name}")
        elif len(seg) < stride:
            # 如果数据不足，填充零
            seg_padded = np.zeros(stride, dtype=np.float32)
            seg_padded[:len(seg)] = seg[:stride]
            seg = seg_padded
        elif len(seg) > stride:
            # 如果数据过多，截断
            seg = seg[:stride]
        out.append(seg.astype(np.float32))
    
    return np.concatenate(out, axis=0).astype(np.float32)


def build_model_obs(
    images: Dict[str, np.ndarray],
    qpos: np.ndarray,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据配置的 model_obs_map，从本地文件构建模型期望的观察格式
    """
    image_cfg = get_image_config(mapping)
    
    # 获取模型观察映射配置
    model_obs_map = mapping.get("model_obs_map", {})
    model_obs = {}
    
    # 处理图像类型的观察
    for model_key, image_name in model_obs_map.items():
        if model_key.startswith("observation.images."):
            # 从 images 字典获取图像
            rgb = images.get(image_name)
            
            # 处理图像并获取嵌套结构
            nested = process_image_obs(rgb, model_key, image_cfg)
            
            # 合并到 model_obs
            merge_nested_dict(model_obs, nested)
        
        elif model_key == "observation.state":
            # 状态数据：根据配置构建多臂状态
            model_arm_order = mapping.get("model_arm_order", [])
            model_arm_dof = int(mapping.get("model_arm_dof", 6))
            use_gripper = bool(mapping.get("use_gripper", True))
            fill_missing_arm = bool(mapping.get("fill_missing_arm", True))
            
            if model_arm_order:
                # 多臂模式：按 model_arm_order 分割和重组
                state_value = build_state_from_qpos(
                    qpos=qpos,
                    model_arm_order=model_arm_order,
                    model_arm_dof=model_arm_dof,
                    use_gripper=use_gripper,
                    fill_missing_arm=fill_missing_arm,
                )
            else:
                # 单臂模式：使用简单的构建方式
                state_value = build_state_obs(
                    qpos=qpos,
                    model_arm_dof=model_arm_dof,
                    use_gripper=use_gripper,
                )
            
            # 将 observation.state 转换为嵌套结构
            if "observation" not in model_obs:
                model_obs["observation"] = {}
            model_obs["observation"]["state"] = state_value
    
    return model_obs


def load_data(data_dir: Path, mapping: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """从数据目录加载数据
    
    Args:
        data_dir: 数据目录路径
        mapping: 模型映射配置
    
    Returns:
        (images_dict, qpos)
    """
    # 查找文件
    qpos_path = data_dir / "qpos.csv"
    if not qpos_path.exists():
        raise FileNotFoundError(f"qpos.csv not found in {data_dir}")

    # 查找图像文件
    model_obs_map = mapping.get("model_obs_map", {})
    images = {}
    all_image_files = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg"))
    
    for model_key, image_name in model_obs_map.items():
        if model_key.startswith("observation.images."):
            image_files = find_image_files(data_dir, image_name)
            
            if not image_files:
                print(f"Warning: No image files found for {image_name}")
                # 如果找不到匹配的图像，尝试使用第一个可用图像
                if all_image_files:
                    image_files = [all_image_files[0]]
                    print(f"  Using first available image: {all_image_files[0]}")
            
            if image_files:
                # 如果有多个匹配，使用第一个
                img_path = image_files[0]
                images[image_name] = load_image(img_path)
                print(f"Loaded image for {image_name}: {img_path.name}")
            else:
                print(f"Warning: No image found for {image_name}, will use zero image")

    # 加载 qpos
    qpos = load_qpos(qpos_path)
    print(f"Loaded qpos: {qpos}")
    
    return images, qpos


__all__ = [
    "load_image",
    "load_qpos",
    "find_image_files",
    "build_model_obs",
    "load_data",
]

