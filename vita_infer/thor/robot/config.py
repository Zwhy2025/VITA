"""
机器人配置类定义
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


@dataclass
class ArmConfig:
    """单个机械臂配置"""

    name: str
    base_topic: str = ""
    dof: int = 6  # 关节自由度（不含夹爪）


@dataclass
class CameraConfig:
    """单个相机配置"""

    name: str
    base_topic: str = ""
    rgb_enabled: bool = True
    depth_enabled: bool = False


@dataclass
class RobotTopicConfig:
    """机器人完整配置（支持YAML加载）"""

    # 机械臂配置列表
    arms: List[ArmConfig] = field(default_factory=list)
    # 相机配置列表
    cameras: List[CameraConfig] = field(default_factory=list)
    # 缓存大小配置
    action_buffer_size: int = 200
    state_buffer_size: int = 30
    camera_buffer_size: int = 30
    # 数据同步配置
    block_timeout: float = 100.0
    check_interval: float = 0.01
    timestamp_tolerance: float = 0.03
    sync_target: str = "image"  # 同步目标，可选 "qpos" 或 "image"

    @property
    def camera_topics(self) -> Dict[str, str]:
        """兼容旧接口：返回 {cam_name: base_topic}"""
        return {cam.name: cam.base_topic for cam in self.cameras}

    @property
    def arm_topics(self) -> Dict[str, str]:
        """兼容旧接口：返回 {arm_name: base_topic}"""
        return {arm.name: arm.base_topic for arm in self.arms}

    @property
    def arm_dof(self) -> int:
        """兼容旧接口：返回第一个机械臂DOF"""
        if self.arms:
            return self.arms[0].dof
        return 6

    def get_arm_dof(self, arm_name: str) -> int:
        """获取指定机械臂的DOF"""
        for arm in self.arms:
            if arm.name == arm_name:
                return arm.dof
        return 6

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "RobotTopicConfig":
        """从YAML配置文件加载配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotTopicConfig":
        """从字典加载配置"""
        # 解析机械臂配置
        arms = []
        arms_data = data.get("arms", {})
        for arm_name, arm_cfg in arms_data.items():
            if isinstance(arm_cfg, dict):
                arms.append(
                    ArmConfig(
                        name=arm_name,
                        base_topic=arm_cfg.get("base_topic", ""),
                        dof=arm_cfg.get("dof", 6),
                    )
                )

        # 解析相机配置
        cameras = []
        cameras_data = data.get("cameras", {})
        for cam_name, cam_cfg in cameras_data.items():
            if isinstance(cam_cfg, dict):
                streams = cam_cfg.get("streams", {})
                cameras.append(
                    CameraConfig(
                        name=cam_name,
                        base_topic=cam_cfg.get("base_topic", ""),
                        rgb_enabled=streams.get("rgb", True),
                        depth_enabled=streams.get("depth", False),
                    )
                )

        # 解析缓存配置
        buffers: dict = data.get("buffers", {})
        action_buffer_size = buffers.get("action_buffer_size", 200)
        state_buffer_size = buffers.get("state_buffer_size", 30)
        camera_buffer_size = buffers.get("camera_buffer_size", 30)

        # 解析同步配置
        sync: dict = data.get("sync", {})
        block_timeout = sync.get("block_timeout", 100.0)
        check_interval = sync.get("check_interval", 0.01)
        timestamp_tolerance = sync.get("timestamp_tolerance", 0.03)
        sync_target = sync.get("sync_target", "image")

        return cls(
            arms=arms,
            cameras=cameras,
            action_buffer_size=action_buffer_size,
            state_buffer_size=state_buffer_size,
            camera_buffer_size=camera_buffer_size,
            block_timeout=block_timeout,
            check_interval=check_interval,
            timestamp_tolerance=timestamp_tolerance,
            sync_target=sync_target,
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
                errors.append(f"机械臂 '{arm.name}' DOF 必须大于 0")

        for cam in self.cameras:
            if not cam.base_topic:
                errors.append(f"相机 '{cam.name}' 缺少 base_topic")

        return errors


__all__ = ["ArmConfig", "CameraConfig", "RobotTopicConfig"]

