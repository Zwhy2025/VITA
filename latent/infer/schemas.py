"""
机器人配置类定义
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ArmConfig:
    """单个机械臂配置"""

    name: str
    base_topic: str
    dof: int


@dataclass
class ImageConfig:
    """图像预处理配置"""

    expected_size: Tuple[int, int]
    normalize: bool
    resize: bool


@dataclass
class SyncConfig:
    """数据同步配置"""

    block_timeout: float
    check_interval: float
    timestamp_tolerance: float
    sync_target: str  # "image" 或 "qpos"


@dataclass
class RobotConfig:
    """机器人完整配置

    arms 的声明顺序即为模型 state/action 向量中的臂顺序。
    cameras 直接映射 model_image_key → topic。
    """

    arms: List[ArmConfig]
    cameras: Dict[str, str]  # model_image_key → topic
    image: ImageConfig
    sync: SyncConfig
    action_delta_threshold: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """从 Hydra 解析后的 dict 构建配置

        期望格式：
            link:
              arms:
                right_arm:
                  base_topic: "..."
                  dof: 6
              cameras:
                image: "topic_url"
            image:
              expected_size: [320, 240]
            sync:
              block_timeout: 100.0
            safety:
              action_delta_threshold: 0.1
        """
        for section_name in ("link", "image", "sync", "safety"):
            if section_name not in data:
                raise ValueError(f"缺少必需配置字段: {section_name}")

        link_data = data["link"]
        for section_name in ("arms", "cameras"):
            if section_name not in link_data:
                raise ValueError(f"link 缺少必需配置字段: {section_name}")

        arms_data = link_data["arms"]
        cameras = dict(link_data["cameras"])
        img_data = data["image"]
        sync_data = data["sync"]
        safety_data = data["safety"]

        # 解析 arms（保留声明顺序）
        arms = []
        for arm_name, arm_cfg in arms_data.items():
            if "topic" in arm_cfg:
                raise ValueError(
                    f"机械臂 '{arm_name}' 使用了已废弃字段 'topic'，请改为 'base_topic'"
                )
            if "base_topic" not in arm_cfg:
                raise ValueError(f"机械臂 '{arm_name}' 必须配置 base_topic")
            if "dof" not in arm_cfg:
                raise ValueError(f"机械臂 '{arm_name}' 必须配置 dof")
            arms.append(
                ArmConfig(
                    name=arm_name,
                    base_topic=str(arm_cfg["base_topic"]),
                    dof=int(arm_cfg["dof"]),
                )
            )

        # 解析 cameras（model_key → topic 直接映射）
        image = ImageConfig(
            expected_size=tuple(img_data["expected_size"]),
            normalize=bool(img_data["normalize"]),
            resize=bool(img_data["resize"]),
        )

        # 解析同步配置
        sync = SyncConfig(
            block_timeout=float(sync_data["block_timeout"]),
            check_interval=float(sync_data["check_interval"]),
            timestamp_tolerance=float(sync_data["timestamp_tolerance"]),
            sync_target=str(sync_data["sync_target"]),
        )

        # 解析安全配置
        action_delta_threshold = float(safety_data["action_delta_threshold"])

        return cls(
            arms=arms,
            cameras=cameras,
            image=image,
            sync=sync,
            action_delta_threshold=action_delta_threshold,
        )

    def validate(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        if not self.arms:
            errors.append("至少需要配置一个机械臂")
        if not self.cameras:
            errors.append("至少需要配置一个相机")

        for arm in self.arms:
            if not arm.base_topic:
                errors.append(f"机械臂 '{arm.name}' 缺少 base_topic")
            if arm.dof <= 0:
                errors.append(f"机械臂 '{arm.name}' dof 必须大于 0")

        for model_key, camera_topic in self.cameras.items():
            if not camera_topic:
                errors.append(f"相机 '{model_key}' 缺少 topic")

        return errors


__all__ = ["ArmConfig", "ImageConfig", "SyncConfig", "RobotConfig"]
