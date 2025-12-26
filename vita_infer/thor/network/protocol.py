"""
网络通信协议
JSON 编解码，支持 numpy 数组序列化
"""
import base64
import json
from typing import Any, Dict

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _object_hook(obj: Dict[str, Any]) -> Any:
    if "__ndarray__" in obj:
        raw = base64.b64decode(obj["data"])
        return np.frombuffer(raw, dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, cls=NumpyJSONEncoder)


def loads(raw: str) -> Any:
    return json.loads(raw, object_hook=_object_hook)


def recv_exact(sock, size: int) -> bytes:
    chunks = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_json(sock) -> Any:
    header = recv_exact(sock, 4)
    length = int.from_bytes(header, "big")
    payload = recv_exact(sock, length).decode("utf-8")
    return loads(payload)


def send_json(sock, payload: Dict[str, Any]) -> None:
    data = dumps(payload).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "big"))
    sock.sendall(data)


__all__ = ["dumps", "loads", "recv_json", "send_json", "NumpyJSONEncoder"]

