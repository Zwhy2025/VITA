"""
VITA 推理客户端主入口
支持在线（连接机器人）和离线（从文件读取）两种模式
"""
import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from thor.utils.config import load_config, merge_configs
from thor.network.client import ModelClient
from thor.io.file import load_data, build_model_obs as build_model_obs_file


def run_offline_mode(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """运行离线模式"""
    client_cfg = cfg.get("client", {})
    mapping = cfg.get("mapping", {})
    
    # 处理 offline_dir 参数
    # 如果指定了场景名，自动匹配 config/data/<scene_name> 目录
    if args.scene and not args.offline_dir:
        # 自动匹配场景对应的数据目录
        offline_dir = Path(__file__).with_name("config") / "data" / args.scene
    elif args.offline_dir:
        offline_dir = Path(args.offline_dir)
        # 如果是相对路径，尝试相对于 config/data 目录
        if not offline_dir.is_absolute():
            base_data_dir = Path(__file__).with_name("config") / "data"
            offline_dir = base_data_dir / offline_dir
    else:
        raise ValueError("离线模式需要指定 --scene 或 --offline-dir 参数")

    if not offline_dir.exists():
        raise FileNotFoundError(f"Offline directory not found: {offline_dir}")
    
    if not offline_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {offline_dir}")

    # 加载离线数据
    images, qpos = load_data(offline_dir, mapping)

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
        model_obs = build_model_obs_file(images, qpos, mapping)
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


def run_online_mode(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """运行在线模式"""
    # 延迟导入机器人相关模块（仅在在线模式需要）
    from thor.io.robot import run_inference_loop
    from thor.robot.center import InteractionDataCenter
    from thor.robot.config import RobotTopicConfig
    
    client_cfg = cfg.get("client", {})
    mapping = cfg.get("mapping", {})

    # 从场景配置中提取 datacenter 配置
    datacenter_cfg = cfg.get("datacenter", {})
    if not datacenter_cfg:
        raise ValueError("datacenter configuration is required for online mode. Please provide it in scene config.")
    
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

        run_inference_loop(center, client, mapping, client_cfg)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VITA Inference Client (支持在线和离线模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 在线模式（连接机器人）
  python vita_client.py --scene ur12e_real_libero_spatial

  # 离线模式（从文件读取）
  python vita_client.py --mode offline --scene ur12e_real_libero_spatial --offline-dir ur12e_real_libero_spatial
  
  # 如果指定了 --offline-dir，自动使用离线模式
  python vita_client.py --offline-dir ur12e_real_libero_spatial
        """,
    )
    parser.add_argument(
        "--global-config",
        type=str,
        default=str(Path(__file__).with_name("config") / "global.yaml"),
        help="全局配置文件路径",
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="场景名称（对应 config/scenes/ 目录下的配置文件）",
    )
    parser.add_argument(
        "--scene-config",
        type=str,
        help="场景配置文件路径（如果指定，将覆盖 --scene 参数）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline"],
        help="运行模式：online（在线，连接机器人）或 offline（离线，从文件读取）。如果指定了 --offline-dir，此参数可省略",
    )
    parser.add_argument(
        "--offline-dir",
        type=str,
        help="离线数据目录路径或子目录名（相对于 config/data 目录）。如果指定了 --scene，会自动匹配 config/data/<scene_name>。如果指定此参数，将自动使用离线模式",
    )
    args = parser.parse_args()

    # 确定运行模式
    if args.offline_dir:
        mode = "offline"
    elif args.mode == "offline":
        mode = "offline"
        # 离线模式需要场景名或离线目录
        if not args.scene and not args.offline_dir:
            parser.error("离线模式需要指定 --scene 或 --offline-dir 参数")
    elif args.mode:
        mode = args.mode
    else:
        mode = "online"  # 默认在线模式

    # 加载全局配置
    global_cfg = load_config(args.global_config)
    
    # 加载场景配置
    scene_cfg = None
    if args.scene_config:
        scene_cfg = load_config(args.scene_config)
    elif args.scene:
        scene_config_path = Path(__file__).with_name("config") / "scenes" / f"{args.scene}.yaml"
        if scene_config_path.exists():
            scene_cfg = load_config(str(scene_config_path))
        else:
            print(f"Warning: Scene config not found: {scene_config_path}, using default mapping")
    
    # 合并配置
    cfg = merge_configs(global_cfg, scene_cfg)

    print(f"Running in {mode} mode")
    
    # 根据模式运行
    if mode == "offline":
        run_offline_mode(args, cfg)
    else:
        run_online_mode(args, cfg)


if __name__ == "__main__":
    main()
