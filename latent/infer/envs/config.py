"""
机器人配置类定义
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ArmConfig:
    """单个机械臂配置"""

    name: str
    topic: str
    dof: int


@dataclass
class ImageConfig:
    """图像预处理配置"""

    expected_size: Tuple[int, int] = (320, 240)
    normalize: bool = True
    resize: bool = False


@dataclass
class SyncConfig:
    """数据同步配置"""

    block_timeout: float = 100.0
    check_interval: float = 0.01
    timestamp_tolerance: float = 0.03
    sync_target: str = "image"  # "image" 或 "qpos"


@dataclass
class RobotConfig:
    """机器人完整配置

    arms 的声明顺序即为模型 state/action 向量中的臂顺序。
    cameras 直接映射 model_image_key → topic。
    """

    arms: List[ArmConfig]
    cameras: Dict[str, str]  # model_image_key → topic
    image: ImageConfig = field(default_factory=ImageConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    action_delta_threshold: float = 0.1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """从 Hydra 解析后的 dict 构建配置

        期望格式：
            link:
              arms:
                right_arm:
                  topic: "..."
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
        link_data = data.get("link", {})

        # 解析 arms（保留声明顺序）
        arms = []
        arms_data = link_data.get("arms", {})
        for arm_name, arm_cfg in arms_data.items():
            if isinstance(arm_cfg, dict):
                if "dof" not in arm_cfg:
                    raise ValueError(f"机械臂 '{arm_name}' 必须配置 dof")
                arms.append(ArmConfig(
                    name=arm_name,
                    topic=arm_cfg.get("topic", ""),
                    dof=int(arm_cfg["dof"]),
                ))

        # 解析 cameras（model_key → topic 直接映射）
        cameras = dict(link_data.get("cameras", {}))

        # 解析图像配置
        img_data = data.get("image", {})
        image = ImageConfig(
            expected_size=tuple(img_data.get("expected_size", [320, 240])),
            normalize=bool(img_data.get("normalize", True)),
            resize=bool(img_data.get("resize", False)),
        )

        # 解析同步配置
        sync_data = data.get("sync", {})
        sync = SyncConfig(
            block_timeout=float(sync_data.get("block_timeout", 100.0)),
            check_interval=float(sync_data.get("check_interval", 0.01)),
            timestamp_tolerance=float(sync_data.get("timestamp_tolerance", 0.03)),
            sync_target=str(sync_data.get("sync_target", "image")),
        )

        # 解析安全配置
        safety_data = data.get("safety", {})
        action_delta_threshold = float(safety_data.get("action_delta_threshold", 0.1))

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
            if not arm.topic:
                errors.append(f"机械臂 '{arm.name}' 缺少 topic")
            if arm.dof <= 0:
                errors.append(f"机械臂 '{arm.name}' dof 必须大于 0")

        for model_key, topic in self.cameras.items():
            if not topic:
                errors.append(f"相机 '{model_key}' 缺少 topic")

        return errors


__all__ = ["ArmConfig", "ImageConfig", "SyncConfig", "RobotConfig"]
