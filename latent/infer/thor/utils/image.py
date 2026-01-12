"""
图像处理工具函数
"""
from typing import Optional

import numpy as np


def resize_image(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """调整图像尺寸"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for resize=true") from exc
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def process_image(
    rgb: Optional[np.ndarray],
    expected_size: Optional[tuple] = None,
    channel_order: str = "rgb",
    normalize: bool = True,
    resize: bool = False,
) -> np.ndarray:
    """处理图像：调整尺寸、转换通道顺序、归一化
    
    Args:
        rgb: RGB 图像数组 (H, W, 3)，可以为 None
        expected_size: 期望尺寸 (width, height)
        channel_order: 通道顺序 "rgb" 或 "bgr"
        normalize: 是否归一化到 [0, 1]
        resize: 尺寸不匹配时是否缩放
    
    Returns:
        处理后的图像数组 (C, H, W)
    """
    expected_w = expected_size[0] if expected_size else None
    expected_h = expected_size[1] if expected_size else None
    
    if rgb is None:
        if expected_w is None or expected_h is None:
            raise RuntimeError("expected_size is required when image is missing")
        rgb = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
    else:
        if expected_w is not None and expected_h is not None:
            if (rgb.shape[1] != expected_w) or (rgb.shape[0] != expected_h):
                if resize:
                    rgb = resize_image(rgb, expected_w, expected_h)
                else:
                    raise RuntimeError(
                        f"image size {rgb.shape[1]}x{rgb.shape[0]} does not match expected {expected_w}x{expected_h}"
                    )
    
    if channel_order == "bgr":
        rgb = rgb[:, :, ::-1]
    
    rgb = rgb.astype(np.float32)
    if normalize:
        rgb = rgb / 255.0
    
    # 转换为 CHW 格式
    chw = np.moveaxis(rgb, -1, 0)
    return chw


__all__ = ["resize_image", "process_image"]

