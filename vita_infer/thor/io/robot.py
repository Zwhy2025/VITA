"""
机器人数据源
连接机器人获取观测数据
"""
import time
from typing import Any, Dict, List

import numpy as np

from thor.io.obs_builder import get_image_config, process_image_obs
from thor.utils.config import merge_nested_dict
from thor.robot.center import InteractionDataCenter


def get_camera_image(images: Dict[str, Any], cam_name: str) -> Any:
    """从 datacenter 获取相机图像"""
    cam = images.get(cam_name)
    if not cam:
        return None
    return cam.get("rgb_img")


def build_agent_pos(
    qpos: np.ndarray,
    dc_arms: List[Any],
    model_arm_order: List[str],
    model_arm_dof: int,
    use_gripper: bool,
    fill_missing_arm: bool,
) -> np.ndarray:
    """构建机器人状态向量"""
    segments: Dict[str, np.ndarray] = {}
    offset = 0
    for arm in dc_arms:
        seg_len = arm.dof + (1 if use_gripper else 0)
        if offset + seg_len <= len(qpos):
            segments[arm.name] = qpos[offset : offset + seg_len]
        else:
            segments[arm.name] = qpos[offset:]
        offset += seg_len

    stride = model_arm_dof + (1 if use_gripper else 0)
    out = []
    for name in model_arm_order:
        seg = segments.get(name)
        if seg is None:
            if fill_missing_arm:
                seg = np.zeros(stride, dtype=np.float32)
            else:
                raise RuntimeError(f"missing arm in observation: {name}")
        if len(seg) < stride:
            seg = np.concatenate([seg, np.zeros(stride - len(seg), dtype=np.float32)])
        out.append(seg[:stride])
    return np.concatenate(out, axis=0).astype(np.float32)


def build_model_obs(
    obs: Dict[str, Any],
    dc_arms: List[Any],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据配置的 model_obs_map，从 datacenter 获取数据并构建模型期望的观察格式
    """
    images = obs.get("image", {})
    image_cfg = get_image_config(mapping)

    # 获取模型观察映射配置
    model_obs_map = mapping.get("model_obs_map", {})
    model_obs = {}
    
    # 处理图像类型的观察
    for model_key, dc_source in model_obs_map.items():
        if model_key.startswith("observation.images."):
            # 图像数据：从 datacenter 相机获取
            rgb = get_camera_image(images, dc_source) if dc_source else None
            
            # 处理图像并获取嵌套结构
            nested = process_image_obs(rgb, model_key, image_cfg)
            
            # 合并到 model_obs
            merge_nested_dict(model_obs, nested)
        
        elif model_key == "observation.state":
            # 状态数据：从 qpos 构建
            qpos = obs.get("qpos", np.array([])).astype(np.float32)
            state_value = build_agent_pos(
                qpos=qpos,
                dc_arms=dc_arms,
                model_arm_order=mapping.get("model_arm_order", []),
                model_arm_dof=int(mapping.get("model_arm_dof", 6)),
                use_gripper=bool(mapping.get("use_gripper", True)),
                fill_missing_arm=bool(mapping.get("fill_missing_arm", True)),
            )
            # 将 observation.state 转换为嵌套结构
            if "observation" not in model_obs:
                model_obs["observation"] = {}
            model_obs["observation"]["state"] = state_value
    
    return model_obs


def map_action_to_datacenter(
    action: np.ndarray,
    dc_arms: List[Any],
    mapping: Dict[str, Any],
) -> np.ndarray:
    """将模型输出的动作映射到 datacenter 格式"""
    model_arm_order = mapping.get("model_arm_order", [])
    model_arm_dof = int(mapping.get("model_arm_dof", 6))
    use_gripper = bool(mapping.get("use_gripper", True))
    stride = model_arm_dof + (1 if use_gripper else 0)

    segments: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(model_arm_order):
        start = idx * stride
        end = start + stride
        segments[name] = action[start:end]

    out: List[float] = []
    for arm in dc_arms:
        seg = segments.get(arm.name)
        target_len = arm.dof + (1 if use_gripper else 0)
        if seg is None:
            seg = np.zeros(target_len, dtype=np.float32)
        if len(seg) < target_len:
            seg = np.concatenate([seg, np.zeros(target_len - len(seg), dtype=np.float32)])
        out.extend(seg[: arm.dof].tolist())
        if use_gripper:
            gripper = float(seg[arm.dof]) if len(seg) > arm.dof else 0.0
            out.append(gripper)
    return np.array(out, dtype=np.float32)


def run_inference_loop(
    center: InteractionDataCenter,
    client: Any,
    mapping: Dict[str, Any],
    client_cfg: Dict[str, Any],
) -> None:
    """运行推理主循环"""
    max_steps = int(client_cfg.get("max_steps", 1000))
    send_freq = float(client_cfg.get("send_freq", 10))
    
    for step_idx in range(max_steps):
        print(f"\n=== Step {step_idx + 1}/{max_steps} ===")
        input("请确认是否执行推理")
        
        # 获取观测
        obs = center.get_observation()
        if not obs:
            print("未获取到观测数据，跳过...")
            continue

        model_obs = build_model_obs(obs, center.config.arms, mapping)
        
        # 调用推理
        resp = client.call("infer", model_obs)
        action = np.asarray(resp.get("action"))
        
        # 确保 action 是 2D 数组 (action_horizon, action_dim)
        if action.ndim == 1:
            # 如果是 1D，转换为 2D (1, action_dim)
            action = action.reshape(1, -1)
        action_seq = action
        
        print(f"Received action sequence: shape={action_seq.shape}")
        
        # 执行动作序列
        for step_idx_in_seq, action_h in enumerate(action_seq):
            dc_action = map_action_to_datacenter(action_h, center.config.arms, mapping)
            
            print(f"  Step {step_idx_in_seq + 1}/{len(action_seq)}: Publishing action")
            input("请确认是否执行推理")
            center.publish_action(dc_action)
            
            time.sleep(1.0 / send_freq if send_freq > 0 else 0)

        print(f"step {step_idx + 1}/{max_steps} completed, action_seq_shape={action_seq.shape}")


__all__ = [
    "get_camera_image",
    "build_agent_pos",
    "build_model_obs",
    "map_action_to_datacenter",
    "run_inference_loop",
]

