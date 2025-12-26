#!/usr/bin/env python3
"""
# @ 版权所有: 2025 Openmind
# @ 作者: hawcat @yanghaotian@efort.com.cn
# @ 创建时间: 2025-12-02 20:16:16
# @ 修改时间: 2025-12-10
# @ 说明: 真机部署话题封装(重构) - 支持YAML配置
"""

import logging
from re import A
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import atlas
import link
import numpy as np
import yaml
from google.protobuf.json_format import MessageToDict
from link._link import SImage, Node, RawPublisher, SubscriberBase
from manip_shared_msg.base.effector_pb2 import EffectorCommand
from manip_shared_msg.base.gripper_pb2 import AdvancedCommand
from manip_shared_msg.base.joint_pb2 import Joints
from manip_shared_msg.locomotion.robot_state_pb2 import RobotState
from manip_shared_msg.locomotion.servo_effector_pb2 import ServoEffector
from manip_shared_msg.locomotion.servo_joint_pb2 import ServoJoint

# ======================== 全局配置 & 日志 ========================
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("robot_interaction")


# ======================== 配置类定义 ========================
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


# ======================== 基础节点抽象类 ========================
class BaseNode(ABC):
    """抽象基类(封装link库节点)"""

    def __init__(self, node_name: str):
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{node_name}]")
        self.node_name = node_name
        self.node_cpp: Optional[Node] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._init_node()

    def _init_node(self):
        """初始化C++节点"""
        with self._lock:
            if not self.node_cpp:
                link.Node.Initialize(self.node_name)
                self.node_cpp = link.GetNode()

    def start_spin(self):
        """启动节点线程(非阻塞)"""
        with self._lock:
            if self._is_running:
                self.logger.warning("节点已在运行中")
                return
            self._is_running = True
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()
            self.logger.info("节点线程已启动")

    def _spin_loop(self):
        """节点循环(内部使用)"""
        self.logger.debug("节点循环启动")
        while self._is_running:
            try:
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"{self.node_name} 节点循环异常: {e}", exc_info=True)
                time.sleep(0.1)
        self.logger.debug("节点循环退出")

    def stop(self):
        """停止节点"""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        if self.node_cpp:
            self.node_cpp.Shutdown()
            self.node_cpp = None
        self.logger.info("节点已停止并释放资源")

    @abstractmethod
    def _init_publishers(self):
        pass

    @abstractmethod
    def _init_subscribers(self):
        pass


# ======================== 机械臂关节节点 ========================
class MultiArmJointNode(BaseNode):
    """机械臂关节/夹爪控制节点(支持多臂)"""

    def __init__(self, node_name: str, config: RobotTopicConfig):
        super().__init__(node_name)
        self.config = config
        self.arm_names = list(self.config.arm_topics.keys())
        self._init_buffers()
        self._init_publishers()
        self._init_subscribers()

    def _init_buffers(self):
        """初始化数据缓存(线程安全)"""
        self.joint_states: Dict[str, RobotState] = {arm: {"timestamp": -1, "orig_timestamp": -1, "data": RobotState()} for arm in self.arm_names}
        self.arm_locks: Dict[str, threading.Lock] = {arm: threading.Lock() for arm in self.arm_names}
        self.action_buffers = deque(maxlen=self.config.action_buffer_size)
        self.state_buffers = deque(maxlen=self.config.state_buffer_size)
        self.logger.info("机械臂缓存初始化完成")

    def _init_publishers(self):
        """初始化机械臂/夹爪发布器"""
        self.arm_publishers: Dict[str, RawPublisher] = {}
        self.gripper_publishers: Dict[str, RawPublisher] = {}

        for arm_name, base_topic in self.config.arm_topics.items():
            joint_topic = f"{base_topic}/joint/servo"
            self.arm_publishers[arm_name] = self.node_cpp.CreatePublisher(joint_topic, ServoJoint)
            gripper_topic = f"{base_topic}/gripper/servo"
            self.gripper_publishers[arm_name] = self.node_cpp.CreatePublisher(gripper_topic, ServoEffector)

        self.logger.info(f"机械臂发布器初始化完成: {self.arm_names}")

    def _init_subscribers(self):
        """初始化关节状态订阅器"""
        self.arm_subscribers: Dict[str, SubscriberBase] = {}

        for arm_name, base_topic in self.config.arm_topics.items():
            state_topic = f"{base_topic}/robot/state"
            self.arm_subscribers[arm_name] = self.node_cpp.CreateSubscriber(state_topic, lambda msg, arm=arm_name: self._joint_state_callback(msg, arm), RobotState)

        self.logger.info(f"机械臂订阅器初始化完成: {self.arm_names}")

    def _joint_state_callback(self, msg: RobotState, arm_name: str):
        """关节状态回调函数"""
        try:
            eff = msg.effector
            status_field = eff.WhichOneof("status")
            if status_field != "gripper":
                self.logger.warning(f"[{arm_name}] 非夹爪状态: {status_field}")
                return
            gripper_ratio = eff.gripper.motion.ratio

            with self.arm_locks[arm_name]:
                ts_sec = msg.header.timestamp.seconds
                ts_nsec = msg.header.timestamp.nanos
                full_ts = ts_sec + ts_nsec * 1e-9
                self.joint_states[arm_name]["orig_timestamp"] = full_ts
                self.joint_states[arm_name]["timestamp"] = time.time_ns()
                self.joint_states[arm_name]["data"] = msg

            state_data = {"timestamp": time.time_ns(), "arm_name": arm_name, "joints": list(msg.joints.position), "gripper_ratio": float(gripper_ratio), "raw_state": MessageToDict(msg)}
            self.state_buffers.append(state_data)

            self.logger.debug(f"[{arm_name}] 收到关节状态: 关节={list(msg.joints.position)}, 夹爪={gripper_ratio:.2f}")

        except Exception:
            self.logger.error(f"[{arm_name}] 关节状态回调异常", exc_info=True)

    def send_servoj(self, arm_name: str, joints: List[float], gripper_ratio: float, speed: float = 0.5, force: float = 0.5) -> bool:
        """发送关节+夹爪控制指令"""
        if arm_name not in self.arm_names:
            self.logger.error(f"机械臂{arm_name}不存在，可选: {self.arm_names}")
            return False

        # 获取该机械臂的实际 DOF
        arm_dof = self.config.get_arm_dof(arm_name)
        if len(joints) != arm_dof:
            self.logger.error(f"关节数量错误: 期望{arm_dof}个，实际{len(joints)}个")
            return False

        try:
            # 构建关节控制消息
            servo_joint_msg = ServoJoint()
            joints_msg = Joints()
            joints_msg.position.extend(joints)
            servo_joint_msg.target.CopyFrom(joints_msg)

            # 构建夹爪控制消息
            servo_gripper_msg = ServoEffector()
            effector_cmd = EffectorCommand()
            gripper_adv_cmd = AdvancedCommand()
            gripper_adv_cmd.ratio = np.clip(gripper_ratio, 0.0, 1.0)  # 限制范围
            gripper_adv_cmd.speed = speed
            gripper_adv_cmd.force = force
            effector_cmd.adv.CopyFrom(gripper_adv_cmd)
            servo_gripper_msg.command.CopyFrom(effector_cmd)

            with self.arm_locks[arm_name]:
                self.arm_publishers[arm_name].Publish(servo_joint_msg)
                self.gripper_publishers[arm_name].Publish(servo_gripper_msg)

            action_data = {
                "timestamp": time.time_ns(),
                "arm_name": arm_name,
                "joints": joints,
                "gripper_ratio": gripper_ratio,
                "raw_joint_cmd": MessageToDict(servo_joint_msg),
                "raw_gripper_cmd": MessageToDict(servo_gripper_msg),
            }
            self.action_buffers.append(action_data)

            self.logger.debug(f"[{arm_name}] 发送控制指令: 关节={joints}, 夹爪={gripper_ratio:.2f}")
            return True

        except Exception:
            self.logger.error(f"[{arm_name}] 发送控制指令失败", exc_info=True)
            return False

    def get_joint_state(self, arm_name: str) -> Dict[str, Optional[Tuple[List[float], float]]]:
        """获取最新关节状态(线程安全)"""
        if arm_name not in self.arm_names:
            self.logger.error(f"机械臂{arm_name}不存在")
            return None

        with self.arm_locks[arm_name]:
            payload = self.joint_states.get(arm_name)
            state: RobotState = payload["data"]
            ts = payload["timestamp"]
            orig_ts = payload["orig_timestamp"]
            if not state:
                self.logger.warning(f"[{arm_name}] 暂无关节状态数据")
                return None

        joints = list(state.joints.position)
        eff = state.effector
        status_field = eff.WhichOneof("status")
        gripper_ratio = float(eff.gripper.motion.ratio) if status_field == "gripper" else 0.0

        return {"timestamp": ts, "data": (joints, gripper_ratio), "orig_timestamp": orig_ts}


# ======================== 相机节点 ========================
class MultiCameraNode(BaseNode):
    """相机图像采集节点(支持RGB/深度流)"""

    def __init__(self, node_name: str, config: RobotTopicConfig):
        super().__init__(node_name)
        self.config = config
        self.camera_names = list(self.config.camera_topics.keys())
        # 根据配置动态生成每个相机的流类型
        self.camera_stream_types: Dict[str, Dict[str, str]] = {}
        for cam in config.cameras:
            streams = {}
            if cam.rgb_enabled:
                streams["rgb_img"] = "color"
            if cam.depth_enabled:
                streams["depth_img"] = "depth"
            self.camera_stream_types[cam.name] = streams
        # 兼容旧逻辑：全局流类型
        self.stream_types = {"rgb_img": "color", "depth_img": "depth"}

        self._init_buffers()
        self._init_subscribers()

    def _init_publishers(self):
        pass

    def _init_buffers(self):
        """初始化图像缓存(线程安全)"""
        self.images: Dict[str, Dict[str, Optional[np.ndarray]]] = {
            cam: {"timestamp": -1, "orig_timestamp": -1, "data": {stream: None for stream in self.stream_types.keys()}} for cam in self.camera_names
        }
        self.camera_locks: Dict[str, threading.Lock] = {cam: threading.Lock() for cam in self.camera_names}
        self.image_buffers = deque(maxlen=self.config.camera_buffer_size)
        self.logger.info("相机缓存初始化完成")

    def _init_subscribers(self):
        """初始化相机流订阅器"""
        self.camera_subscribers: Dict[str, Dict[str, SubscriberBase]] = {}

        for cam_name, base_topic in self.config.camera_topics.items():
            self.camera_subscribers[cam_name] = {}
            # 获取该相机启用的流类型
            stream_types = self.camera_stream_types.get(cam_name, self.stream_types)
            for stream_name, topic_suffix in stream_types.items():
                stream_topic = f"{base_topic}/{topic_suffix}"
                self.camera_subscribers[cam_name][stream_name] = self.node_cpp.CreateSubscriber(
                    stream_topic, lambda msg, cam=cam_name, stream=stream_name: self._camera_callback(msg, cam, stream), SImage
                )

        self.logger.info(f"相机订阅器初始化完成: {self.camera_names}, 流配置: {self.camera_stream_types}")

    def _camera_callback(self, msg: SImage, cam_name: str, stream_name: str):
        """相机图像回调函数(线程安全)"""
        try:
            # 将protobuf图像转换为numpy数组
            img_np = atlas.utils.image_to_numpy(msg)
            if img_np is None or img_np.size == 0:
                self.logger.warning(f"[{cam_name}/{stream_name}] 空图像数据")
                return

            curr_timestamp = time.time_ns()

            with self.camera_locks[cam_name]:
                ts_sec = msg.header.stamp.sec
                ts_nsec = msg.header.stamp.nanosec
                full_ts = ts_sec + ts_nsec * 1e-9
                self.images[cam_name]["orig_timestamp"] = full_ts
                self.images[cam_name]["timestamp"] = curr_timestamp
                self.images[cam_name]["data"][stream_name] = img_np

            image_data = {"timestamp": curr_timestamp, "camera_name": cam_name, "stream_type": stream_name, "shape": img_np.shape, "dtype": str(img_np.dtype)}
            self.image_buffers.append(image_data)

            self.logger.debug(f"[{cam_name}/{stream_name}] 收到图像: {img_np.shape}")

        except Exception:
            self.logger.error(f"[{cam_name}/{stream_name}] 图像回调异常", exc_info=True)

    def get_camera_img(self, cam_name: str) -> Optional[Dict[str, Optional[np.ndarray]]]:
        """获取指定相机的最新图像(RGB+深度)，不要对返回值进行修改"""
        if cam_name not in self.camera_names:
            self.logger.error(f"相机{cam_name}不存在，可选: {self.camera_names}")
            return None

        with self.camera_locks[cam_name]:
            return self.images[cam_name]


# ======================== 交互中心(组合节点) ========================
class InteractionDataCenter:
    """机器人交互中心(组合关节/相机节点)"""

    def __init__(
        self,
        config: Optional[RobotTopicConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        初始化交互中心

        Args:
            config: 直接传入的配置对象（优先级最高）
            config_path: YAML配置文件路径（config为None时使用）
        """
        # 加载配置（优先级：config > config_path > 默认配置）
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = RobotTopicConfig.from_yaml(config_path)
        else:
            # 默认从 config.yaml 加载，但需要提取 datacenter 部分
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            
            # 从合并后的配置中提取 datacenter 部分
            datacenter_data = full_config.get("datacenter", {})
            if not datacenter_data:
                raise ValueError(f"配置文件 {config_path} 中缺少 'datacenter' 配置节")
            
            self.config = RobotTopicConfig.from_dict(datacenter_data)

        # 验证配置
        errors = self.config.validate()
        if errors:
            raise ValueError(f"{config_path}：配置验证失败: {errors}")

        self.is_running = False

        # 初始化子节点
        self.arm_node = MultiArmJointNode("multi_arm_joint_node", self.config)
        self.camera_node = MultiCameraNode("multi_camera_node", self.config)

        self.logger = logging.getLogger("InteractionDataCenter")

        self.last_obs: Optional[Dict] = None
        self.last_obs_lock = threading.Lock()
        self.last_obs_id: int = 0
        # 从配置读取同步参数
        self.block_timeout: float = self.config.block_timeout
        self.check_interval: float = self.config.check_interval
        self.timestamp_tolerance: float = self.config.timestamp_tolerance

        self.logger.info(f"交互中心初始化完成，启用机械臂: {list(self.config.arm_topics.keys())}, 启用相机: {list(self.config.camera_topics.keys())}")

    def get_camera_names(self) -> Dict[str, str]:
        """获取启用的相机名称列表"""
        return self.config.camera_topics

    def get_arm_names(self) -> Dict[str, str]:
        """获取启用的机械臂名称列表"""
        return self.config.arm_topics

    def _is_obs_updated(self, new_obs: Dict, tolerance: Optional[float] = None) -> bool:
        """基于每个相机/机械臂的原始时间戳判断是否更新。"""
        if self.last_obs is None:
            return True

        # 任意一相机或机械臂时间戳差值大于等于容忍值即视为更新
        # 检查相机原始时间戳
        if self.config.sync_target == "image":
            for cam_name in self.config.camera_topics.keys():
                key = f"orig_image_ts_{cam_name}"
                prev_ts = self.last_obs.get(key)
                curr_ts = new_obs.get(key)
                if prev_ts is None or curr_ts is None:
                    return True
                if curr_ts != prev_ts:
                    return True

        elif self.config.sync_target == "qpos":
            # 检查机械臂原始时间戳
            for arm_name in self.config.arm_topics.keys():
                key = f"orig_qpos_ts_{arm_name}"
                prev_ts = self.last_obs.get(key)
                curr_ts = new_obs.get(key)
                if prev_ts is None or curr_ts is None:
                    return True
                if abs(curr_ts - prev_ts) >= tolerance:
                    return True

        else:
            self.logger.warning(f"未知的 sync_target 配置: {self.config.sync_target}")
            return True

        return False

    def start(self):
        """启动所有节点"""
        if self.is_running:
            self.logger.warning("交互中心已在运行中")
            return

        # 启动节点
        self.arm_node.start_spin()
        self.camera_node.start_spin()
        self.is_running = True
        self.logger.info("交互中心启动完成")

    def stop(self):
        """停止所有节点"""
        if not self.is_running:
            return

        # 停止子节点
        self.arm_node.stop()
        self.camera_node.stop()
        self.is_running = False
        self.logger.info("交互中心已停止")

    def get_observation(self) -> Dict[str, Union[Dict, np.ndarray, str]]:
        """获取完整观测(框架适配)"""
        start_time = time.time()
        logger.info("开始获取观测...")

        while self.is_running:
            # 检查是否超时
            if time.time() - start_time > self.block_timeout:
                self.logger.warning(f"观测获取超时 {self.block_timeout}s，无有效相机/数据未更新")
                return {}

            observation = {}
            images = {}
            image_ts = []
            camera_ok_list = []

            for cam_name in self.camera_node.camera_names:
                payload = self.camera_node.get_camera_img(cam_name)
                img_data = payload.get("data")
                image_ts.append(payload.get("timestamp"))
                observation[f"orig_image_ts_{cam_name}"] = payload.get("orig_timestamp")

                cam_ok = img_data["rgb_img"] is not None and img_data["rgb_img"].size > 0
                camera_ok_list.append(cam_ok)
                images[cam_name] = img_data

            if not all(camera_ok_list):
                time.sleep(self.check_interval)
                continue

            qpos = []
            qpos_ts = []
            for arm_name in self.arm_node.arm_names:
                payload = self.arm_node.get_joint_state(arm_name)
                joint_state = payload.get("data")
                qpos_ts.append(payload.get("timestamp"))
                observation[f"orig_qpos_ts_{arm_name}"] = payload.get("orig_timestamp")

                if joint_state:
                    joints, gripper = joint_state
                    qpos.extend(joints)
                    qpos.append(gripper)

            observation["image"] = images
            observation["image_ts"] = sum(image_ts) / len(image_ts)
            observation["qpos"] = np.array(qpos, dtype=np.float32)
            observation["qpos_ts"] = sum(qpos_ts) / len(qpos_ts)

            with self.last_obs_lock:
                if not self._is_obs_updated(observation, tolerance=self.timestamp_tolerance):
                    logger.debug("观测数据未更新，等待...")
                    time.sleep(self.check_interval)
                    continue

            with self.last_obs_lock:
                self.last_obs = observation.copy()

            self.logger.debug(f"成功获取有效观测：相机有效数={sum(camera_ok_list)}, 关节维度={observation['qpos'].shape}")
            return observation

    def step(self, action: np.ndarray) -> Dict[str, Union[Dict, np.ndarray, str]]:
        """
        执行动作并获取下一个观测(框架适配)

        Args:
            action: 动作数组，按配置中机械臂顺序排列
                    格式：[臂1关节+夹爪, 臂2关节+夹爪, ...]
        """
        success = self.publish_action(action)
        if not success:
            self.logger.error("动作执行失败，无法获取下一个观测")
            return {}

        return self.get_observation()

    def publish_action(self, action: np.ndarray) -> bool:
        """
        发布动作指令(框架适配)

        Args:
            action: 动作数组，按配置中机械臂顺序排列
                    格式：[臂1关节+夹爪, 臂2关节+夹爪, ...]
        """
        if not self.is_running:
            self.logger.error("交互中心未启动，无法发布动作")
            return False

        try:
            offset = 0  # 用于判定当前处理到哪个机械臂
            all_success = True

            for arm in self.config.arms:
                arm_len = arm.dof + 1

                if len(action) < offset + arm_len:
                    self.logger.warning(f"动作数组长度不足，跳过机械臂 '{arm.name}'")
                    break

                arm_action = action[offset : offset + arm_len]
                joints = arm_action[: arm.dof].tolist()
                gripper = float(arm_action[arm.dof])
                

                threshold = 0.35
                if gripper > 1- threshold:
                    gripper = 1.0
                if gripper < threshold:
                    gripper = 0

                success = self.arm_node.send_servoj(arm.name, joints, gripper)
                all_success = all_success and success

                offset += arm_len

            return all_success

        except Exception as e:
            self.logger.error(f"动作发布失败：{e}")
            return False


# ======================== 测试入口 ========================
def test_spin(config_path: Optional[str] = None):
    """测试"""
    center = InteractionDataCenter(config_path=config_path)
    try:
        center.start()
        logger.info("交互中心已启动，按Ctrl+C退出")
        logger.info(f"当前配置：机械臂={list(center.config.arm_topics.keys())}, 相机={list(center.config.camera_topics.keys())}")

        while True:
            obs = center.get_observation()
            if obs:
                logger.info(f"观测获取成功: qpos.shape={obs['qpos'].shape}, 相机数={len(obs['image'])}")
            else:
                logger.warning("观测获取失败或超时")
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("收到退出信号，停止交互中心")
    finally:
        center.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="机器人交互中心测试")
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML配置文件路径")
    args = parser.parse_args()

    test_spin(config_path=args.config)
