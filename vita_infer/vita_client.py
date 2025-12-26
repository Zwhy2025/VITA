"""
VITA 推理客户端
连接 DataCenter 获取观测数据，发送到 VITA Server 进行推理
"""
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from protocol import recv_json, send_json
from datacenter import InteractionDataCenter, RobotTopicConfig


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resize_image(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for resize=true") from exc
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _get_camera_image(images: Dict[str, Any], cam_name: str) -> Optional[np.ndarray]:
    cam = images.get(cam_name)
    if not cam:
        return None
    return cam.get("rgb_img")


def _build_agent_pos(
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


def _build_model_obs(
    obs: Dict[str, Any],
    dc_arms: List[Any],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据配置的 model_obs_map，从 datacenter 获取数据并构建模型期望的观察格式
    """
    images = obs.get("image", {})
    image_cfg = mapping.get("image", {})
    expected_size = image_cfg.get("expected_size", None)
    expected_w = expected_size[0] if expected_size else None
    expected_h = expected_size[1] if expected_size else None
    channel_order = image_cfg.get("channel_order", "rgb")
    normalize = bool(image_cfg.get("normalize", True))
    resize = bool(image_cfg.get("resize", False))

    # 获取模型观察映射配置
    model_obs_map = mapping.get("model_obs_map", {})
    model_obs = {}
    
    # 处理图像类型的观察
    for model_key, dc_source in model_obs_map.items():
        if model_key.startswith("observation.images."):
            # 图像数据：从 datacenter 相机获取
            rgb = _get_camera_image(images, dc_source) if dc_source else None
            if rgb is None:
                if expected_w is None or expected_h is None:
                    raise RuntimeError(f"expected_size is required when camera {dc_source} image is missing")
                rgb = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
            else:
                if expected_w is not None and expected_h is not None:
                    if (rgb.shape[1] != expected_w) or (rgb.shape[0] != expected_h):
                        if resize:
                            rgb = _resize_image(rgb, expected_w, expected_h)
                        else:
                            raise RuntimeError(
                                f"camera {dc_source} size {rgb.shape[1]}x{rgb.shape[0]} does not match expected {expected_w}x{expected_h}"
                            )
            if channel_order == "bgr":
                rgb = rgb[:, :, ::-1]
            rgb = rgb.astype(np.float32)
            if normalize:
                rgb = rgb / 255.0
            chw = np.moveaxis(rgb, -1, 0)
            
            # 将点分隔键转换为嵌套结构
            parts = model_key.split(".")
            current = model_obs
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = chw
        
        elif model_key == "observation.state":
            # 状态数据：从 qpos 构建
            qpos = obs.get("qpos", np.array([])).astype(np.float32)
            state_value = _build_agent_pos(
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


def _map_action_to_datacenter(
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


class ModelClient:
    """模型推理客户端"""
    
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        import socket

        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

    def call(self, cmd: str, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        send_json(self.sock, {"cmd": cmd, "obs": obs})
        resp = recv_json(self.sock)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(resp.get("error"))
        return resp

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def main() -> None:
    parser = argparse.ArgumentParser(description="VITA Inference Client")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    client_cfg = cfg.get("client", {})
    mapping = cfg.get("mapping", {})

    # 从配置中提取 datacenter 配置
    datacenter_cfg = cfg.get("datacenter", {})
    datacenter_config = RobotTopicConfig.from_dict(datacenter_cfg)
    center = InteractionDataCenter(config=datacenter_config)

    log_dir = Path(client_cfg.get("log_dir", "./real_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    client = None
    try:
        center.start()
        client = ModelClient(
            host=client_cfg.get("server_host", "127.0.0.1"),
            port=int(client_cfg.get("server_port", 8548)),
            timeout=float(client_cfg.get("action_timeout", 5.0)),
        )
        client.call("reset")

        max_steps = int(client_cfg.get("max_steps", 1000))
        send_freq = float(client_cfg.get("send_freq", 10))
        
        for step_idx in range(max_steps):
            print(f"\n=== Step {step_idx + 1}/{max_steps} ===")
        
            # input("请确认是否执行推理")
            # 获取观测
            obs = center.get_observation() 
            if not obs:
                print("未获取到观测数据，跳过...")
                continue

            model_obs = _build_model_obs(obs, center.config.arms, mapping)
            
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
                
                # if step_idx_in_seq==0:
                #     continue
    
                dc_action = _map_action_to_datacenter(action_h, center.config.arms, mapping)
                
                print(f"  Step {step_idx_in_seq + 1}/{len(action_seq)}: Publishing action: {dc_action}")
                # input("请确认是否发布动作")
                center.publish_action(dc_action)
                
                time.sleep(1.0 / send_freq if send_freq > 0 else 0)

            print(f"step {step_idx + 1}/{max_steps} completed, action_seq_shape={action_seq.shape}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.close()
        center.stop()


if __name__ == "__main__":
    main()

