"""
VITA 离线推理客户端
从本地文件读取观测数据，发送到 VITA Server 进行推理
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from protocol import recv_json, send_json


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resize_image(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _load_image(image_path: Path) -> np.ndarray:
    """加载图像文件"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_qpos(qpos_path: Path) -> np.ndarray:
    """从 CSV 文件加载 qpos"""
    with open(qpos_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        row = next(reader)  # 读取第一行数据
        # 解析数据，去除空格
        qpos = [float(x.strip()) for x in row]
    return np.array(qpos, dtype=np.float32)


def _build_model_obs(
    images: Dict[str, np.ndarray],
    qpos: np.ndarray,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据配置的 model_obs_map，从本地文件构建模型期望的观察格式
    """
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
    for model_key, image_name in model_obs_map.items():
        if model_key.startswith("observation.images."):
            # 从 images 字典获取图像
            rgb = images.get(image_name)
            if rgb is None:
                if expected_w is None or expected_h is None:
                    raise RuntimeError(f"expected_size is required when image {image_name} is missing")
                rgb = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
            else:
                if expected_w is not None and expected_h is not None:
                    if (rgb.shape[1] != expected_w) or (rgb.shape[0] != expected_h):
                        if resize:
                            rgb = _resize_image(rgb, expected_w, expected_h)
                        else:
                            raise RuntimeError(
                                f"image {image_name} size {rgb.shape[1]}x{rgb.shape[0]} does not match expected {expected_w}x{expected_h}"
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
            # 状态数据：直接使用 qpos
            # 根据配置，模型期望 7 维（6 dof + 1 gripper）
            model_arm_dof = int(mapping.get("model_arm_dof", 6))
            use_gripper = bool(mapping.get("use_gripper", True))
            
            # 确保 qpos 维度正确
            expected_dim = model_arm_dof + (1 if use_gripper else 0)
            if len(qpos) < expected_dim:
                # 填充零
                qpos_padded = np.zeros(expected_dim, dtype=np.float32)
                qpos_padded[:len(qpos)] = qpos[:expected_dim]
                state_value = qpos_padded
            elif len(qpos) > expected_dim:
                # 截断
                state_value = qpos[:expected_dim]
            else:
                state_value = qpos
            
            # 将 observation.state 转换为嵌套结构
            if "observation" not in model_obs:
                model_obs["observation"] = {}
            model_obs["observation"]["state"] = state_value.astype(np.float32)
    
    return model_obs


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
    }
    
    search_names = name_mapping.get(image_name, [image_name])
    
    # 支持的文件扩展名（不使用通配符，直接匹配）
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


def main() -> None:
    parser = argparse.ArgumentParser(description="VITA Offline Inference Client")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--offline-dir", type=str, default=str(Path(__file__).with_name("offline")))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    client_cfg = cfg.get("client", {})
    mapping = cfg.get("mapping", {})
    offline_dir = Path(args.offline_dir)

    if not offline_dir.exists():
        raise FileNotFoundError(f"Offline directory not found: {offline_dir}")

    # 查找文件
    qpos_path = offline_dir / "qpos.csv"
    if not qpos_path.exists():
        raise FileNotFoundError(f"qpos.csv not found in {offline_dir}")

    # 查找图像文件
    model_obs_map = mapping.get("model_obs_map", {})
    images = {}
    all_image_files = list(offline_dir.glob("*.png")) + list(offline_dir.glob("*.jpg")) + list(offline_dir.glob("*.jpeg"))
    
    for model_key, image_name in model_obs_map.items():
        if model_key.startswith("observation.images."):
            image_files = find_image_files(offline_dir, image_name)
            
            if not image_files:
                print(f"Warning: No image files found for {image_name}")
                # 如果找不到匹配的图像，尝试使用第一个可用图像
                if all_image_files:
                    image_files = [all_image_files[0]]
                    print(f"  Using first available image: {all_image_files[0]}")
            
            if image_files:
                # 如果有多个匹配，使用第一个
                img_path = image_files[0]
                images[image_name] = _load_image(img_path)
                print(f"Loaded image for {image_name}: {img_path.name}")
            else:
                print(f"Warning: No image found for {image_name}, will use zero image")

    # 加载 qpos
    qpos = _load_qpos(qpos_path)
    print(f"Loaded qpos: {qpos}")

    client = None
    try:
        client = ModelClient(
            host=client_cfg.get("server_host", "127.0.0.1"),
            port=int(client_cfg.get("server_port", 8548)),
            timeout=float(client_cfg.get("action_timeout", 5.0)),
        )
        
        print("Connected to VITA server")
        client.call("reset")
        print("Reset completed")

        # 构建观测
        model_obs = _build_model_obs(images, qpos, mapping)
        print(f"Built observation with keys: {list(model_obs.keys())}")
        
        # 调用推理
        print("\n=== Sending inference request ===")
        start_time = time.time()
        resp = client.call("infer", model_obs)
        action = np.asarray(resp.get("action"))
        end_time = time.time()
        print(f"Inference time: {end_time - start_time} seconds")
        print(f"\n=== Inference Result ===")
        print(f"Action shape: {action.shape}")
        print(f"Action:\n{action}")
        
        # 如果 action 是序列，打印每一步
        if len(action.shape) == 2:
            print(f"\nAction sequence ({action.shape[0]} steps):")
            for i, act in enumerate(action):
                print(f"  Step {i+1}: {act}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()

