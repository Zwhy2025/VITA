"""
模型推理客户端
"""
import socket
from typing import Any, Dict, Optional

from thor.network.protocol import recv_json, send_json


class ModelClient:
    """模型推理客户端"""
    
    def __init__(self, host: str, port: int, timeout: float = 5.0):
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


__all__ = ["ModelClient"]

