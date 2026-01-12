"""
VITA 推理服务器
基于 TCP Socket 提供推理服务
"""
import argparse
import logging
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml

from thor.network.protocol import recv_json, send_json
from thor.engine.inference import VitaInference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _load_config(path: str) -> Dict[str, Any]:
    logger.info(f"Loading config from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded successfully")
    return cfg


def _extract_all_keys(obj: Any, prefix: str = "") -> list:
    """递归提取字典或嵌套结构中的所有键"""
    keys = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys.append(current_key)
            # 如果值是字典或列表，递归提取
            if isinstance(value, (dict, list)):
                keys.extend(_extract_all_keys(value, current_key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(item, (dict, list)):
                keys.extend(_extract_all_keys(item, current_key))
    return keys


def _build_model(cfg: Dict[str, Any]):
    model_cfg = cfg.get("model", {})
    checkpoint_dir = model_cfg.get("checkpoint_dir")
    mixed_precision = model_cfg.get("mixed_precision", "bf16")
    device = model_cfg.get("device", "cuda")

    logger.info(f"Building VITA model with config:")
    logger.info(f"  checkpoint_dir: {checkpoint_dir}")
    logger.info(f"  mixed_precision: {mixed_precision}")
    logger.info(f"  device: {device}")

    if device == "cpu" and mixed_precision != "fp32":
        logger.warning(f"Device is CPU, forcing mixed_precision to fp32")
        mixed_precision = "fp32"

    logger.info("Loading VITA model checkpoint...")
    start_time = time.time()
    model = VitaInference(
        checkpoint_dir=checkpoint_dir,
        mixed_precision=mixed_precision,
        device=device,
    )
    load_time = time.time() - start_time
    logger.info(f"VITA model loaded successfully in {load_time:.2f} seconds")
    return model


class VitaService:
    """VITA 推理服务"""
    
    def __init__(self, model: VitaInference):
        self.model = model
        self.infer_count = 0
        logger.info("VitaService initialized")

    def reset(self) -> Dict[str, Any]:
        logger.info("Resetting model state")
        self.model.reset()
        self.infer_count = 0
        return {"ok": True}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if obs is None:
            raise ValueError("obs is required")
        
        self.infer_count += 1
        total_start = time.time()
        
        # 更新观测
        update_start = time.time()
        self.model.update_obs(obs)
        logger.debug(f"[TIMING] update_obs: {(time.time() - update_start) * 1000:.2f}ms")
        
        # 获取动作
        get_action_start = time.time()
        action = self.model.get_action()
        logger.debug(f"[TIMING] get_action: {(time.time() - get_action_start) * 1000:.2f}ms")
        
        infer_time = time.time() - total_start
        logger.info(f"Inference #{self.infer_count} completed in {infer_time*1000:.2f}ms")
        
        return {"action": action}

    def ping(self) -> Dict[str, Any]:
        logger.debug("Ping received")
        return {"ok": True}


class ModelServer:
    """TCP 模型服务器"""
    
    def __init__(self, host: str, port: int, service: VitaService):
        self.host = host
        self.port = port
        self.service = service
        self.sock = None
        self.running = False
        self.threads = []

    def start(self) -> None:
        logger.info(f"Starting VITA server on {self.host}:{self.port}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True
        logger.info(f"VITA server listening on {self.host}:{self.port}")

        try:
            while self.running:
                logger.info("Waiting for client connection...")
                client, addr = self.sock.accept()
                logger.info(f"Client connected from {addr[0]}:{addr[1]}")
                thread = threading.Thread(target=self._handle_client, args=(client, addr), daemon=True)
                thread.start()
                self.threads.append(thread)
                logger.info(f"Active client threads: {len(self.threads)}")
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.stop()

    def stop(self) -> None:
        logger.info("Stopping server...")
        self.running = False
        if self.sock:
            try:
                self.sock.close()
                logger.info("Socket closed")
            except OSError as e:
                logger.warning(f"Error closing socket: {e}")
        logger.info(f"Waiting for {len(self.threads)} client threads to finish...")
        for t in self.threads:
            t.join(timeout=1)
        logger.info("Server stopped")

    def _handle_client(self, client_sock: socket.socket, addr: tuple) -> None:
        client_id = f"{addr[0]}:{addr[1]}"
        logger.info(f"[{client_id}] Starting client handler thread")
        
        with client_sock:
            while self.running:
                try:
                    logger.debug(f"[{client_id}] Waiting for request...")
                    data = recv_json(client_sock)
                    cmd = data.get("cmd")
                    obs = data.get("obs")

                    logger.info(f"[{client_id}] Received command: {cmd}")

                    if cmd == "reset":
                        resp = self.service.reset()
                        logger.info(f"[{client_id}] Reset completed")
                    elif cmd == "infer":
                        
                        # TODO 调试代码
                        import json
                        obs_keys = list(obs.keys()) if obs else []
                        all_keys = _extract_all_keys(obs)
                        all_json_keys = json.dumps(all_keys, indent=2)
                        print(f"[{client_id}] Inference request with obs keys: {all_json_keys}")
                        
                        resp = self.service.infer(obs)
                        action = resp.get("action")
                        action_shape = action.shape if hasattr(action, "shape") else "unknown"
                        logger.debug(f"[{client_id}] Inference completed, action shape: {action_shape}")
                    elif cmd == "ping":
                        resp = self.service.ping()
                        logger.debug(f"[{client_id}] Ping responded")
                    else:
                        logger.warning(f"[{client_id}] Unknown command: {cmd}")
                        raise ValueError(f"unknown cmd: {cmd}")

                    send_json(client_sock, resp)
                    logger.debug(f"[{client_id}] Response sent")
                except ConnectionError as e:
                    logger.info(f"[{client_id}] Client disconnected: {e}")
                    break
                except Exception as exc:
                    err = f"server error: {exc}"
                    tb = traceback.format_exc()
                    logger.error(f"[{client_id}] Error handling request: {err}")
                    logger.error(f"[{client_id}] Traceback:\n{tb}")
                    send_json(client_sock, {"error": err, "traceback": tb})
                    break
        
        logger.info(f"[{client_id}] Client handler thread ended")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting VITA Inference Server")
    logger.info("=" * 60)
    
    parser = argparse.ArgumentParser(description="VITA Inference Server")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config") / "global.yaml"),
                       help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="DEBUG",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Log level set to: {args.log_level}")

    cfg = _load_config(args.config)
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8548))
    
    logger.info(f"Server configuration: host={host}, port={port}")

    model = _build_model(cfg)
    service = VitaService(model)
    server = ModelServer(host, port, service)
    
    try:
        server.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
