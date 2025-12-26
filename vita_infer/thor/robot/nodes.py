"""
机器人节点类定义
包含基础节点抽象类和机械臂/相机节点实现
"""
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from thor.robot.config import RobotTopicConfig

# 类型提示时导入，避免循环导入
if TYPE_CHECKING:
    from link._link import Node, RawPublisher, SubscriberBase, SImage
    from manip_shared_msg.locomotion.robot_state_pb2 import RobotState


def _import_robot_dependencies():
    """延迟导入机器人相关依赖（仅在需要时导入）"""
    import atlas
    import link
    from google.protobuf.json_format import MessageToDict
    from link._link import SImage, Node, RawPublisher, SubscriberBase
    from manip_shared_msg.base.effector_pb2 import EffectorCommand
    from manip_shared_msg.base.gripper_pb2 import AdvancedCommand
    from manip_shared_msg.base.joint_pb2 import Joints
    from manip_shared_msg.locomotion.robot_state_pb2 import RobotState
    from manip_shared_msg.locomotion.servo_effector_pb2 import ServoEffector
    from manip_shared_msg.locomotion.servo_joint_pb2 import ServoJoint
    
    return {
        'atlas': atlas,
        'link': link,
        'MessageToDict': MessageToDict,
        'SImage': SImage,
        'Node': Node,
        'RawPublisher': RawPublisher,
        'SubscriberBase': SubscriberBase,
        'EffectorCommand': EffectorCommand,
        'AdvancedCommand': AdvancedCommand,
        'Joints': Joints,
        'RobotState': RobotState,
        'ServoEffector': ServoEffector,
        'ServoJoint': ServoJoint,
    }


class BaseNode(ABC):
    """抽象基类(封装link库节点)"""

    def __init__(self, node_name: str):
        # 延迟导入依赖
        deps = _import_robot_dependencies()
        self._link = deps['link']
        self._Node = deps['Node']
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{node_name}]")
        self.node_name = node_name
        self.node_cpp: Optional[Any] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._init_node()

    def _init_node(self):
        """初始化C++节点"""
        with self._lock:
            if not self.node_cpp:
                self._link.Node.Initialize(self.node_name)
                self.node_cpp = self._link.GetNode()

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


class MultiArmJointNode(BaseNode):
    """机械臂关节/夹爪控制节点(支持多臂)"""

    def __init__(self, node_name: str, config: RobotTopicConfig):
        # 延迟导入依赖
        self._deps = _import_robot_dependencies()
        self._RobotState = self._deps['RobotState']
        self._ServoJoint = self._deps['ServoJoint']
        self._ServoEffector = self._deps['ServoEffector']
        self._EffectorCommand = self._deps['EffectorCommand']
        self._AdvancedCommand = self._deps['AdvancedCommand']
        self._Joints = self._deps['Joints']
        self._MessageToDict = self._deps['MessageToDict']
        self._RawPublisher = self._deps['RawPublisher']
        self._SubscriberBase = self._deps['SubscriberBase']
        
        super().__init__(node_name)
        self.config = config
        self.arm_names = list(self.config.arm_topics.keys())
        self._init_buffers()
        self._init_publishers()
        self._init_subscribers()

    def _init_buffers(self):
        """初始化数据缓存(线程安全)"""
        RobotState = self._RobotState
        self.joint_states: Dict[str, Dict] = {arm: {"timestamp": -1, "orig_timestamp": -1, "data": RobotState()} for arm in self.arm_names}
        self.arm_locks: Dict[str, threading.Lock] = {arm: threading.Lock() for arm in self.arm_names}
        self.action_buffers = deque(maxlen=self.config.action_buffer_size)
        self.state_buffers = deque(maxlen=self.config.state_buffer_size)
        self.logger.info("机械臂缓存初始化完成")

    def _init_publishers(self):
        """初始化机械臂/夹爪发布器"""
        self.arm_publishers: Dict[str, Any] = {}
        self.gripper_publishers: Dict[str, Any] = {}
        ServoJoint = self._ServoJoint
        ServoEffector = self._ServoEffector

        for arm_name, base_topic in self.config.arm_topics.items():
            joint_topic = f"{base_topic}/joint/servo"
            self.arm_publishers[arm_name] = self.node_cpp.CreatePublisher(joint_topic, ServoJoint)
            gripper_topic = f"{base_topic}/gripper/servo"
            self.gripper_publishers[arm_name] = self.node_cpp.CreatePublisher(gripper_topic, ServoEffector)

        self.logger.info(f"机械臂发布器初始化完成: {self.arm_names}")

    def _init_subscribers(self):
        """初始化关节状态订阅器"""
        self.arm_subscribers: Dict[str, Any] = {}
        RobotState = self._RobotState

        for arm_name, base_topic in self.config.arm_topics.items():
            state_topic = f"{base_topic}/robot/state"
            self.arm_subscribers[arm_name] = self.node_cpp.CreateSubscriber(
                state_topic, lambda msg, arm=arm_name: self._joint_state_callback(msg, arm), RobotState
            )

        self.logger.info(f"机械臂订阅器初始化完成: {self.arm_names}")

    def _joint_state_callback(self, msg, arm_name: str):
        """关节状态回调函数"""
        MessageToDict = self._MessageToDict
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
        ServoJoint = self._ServoJoint
        ServoEffector = self._ServoEffector
        EffectorCommand = self._EffectorCommand
        AdvancedCommand = self._AdvancedCommand
        Joints = self._Joints
        MessageToDict = self._MessageToDict
        
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
            state = payload["data"]
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


class MultiCameraNode(BaseNode):
    """相机图像采集节点(支持RGB/深度流)"""

    def __init__(self, node_name: str, config: RobotTopicConfig):
        # 延迟导入依赖
        self._deps = _import_robot_dependencies()
        self._atlas = self._deps['atlas']
        self._SImage = self._deps['SImage']
        self._SubscriberBase = self._deps['SubscriberBase']
        
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
        self.camera_subscribers: Dict[str, Dict[str, Any]] = {}
        SImage = self._SImage

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

    def _camera_callback(self, msg, cam_name: str, stream_name: str):
        """相机图像回调函数(线程安全)"""
        atlas = self._atlas
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


__all__ = ["BaseNode", "MultiArmJointNode", "MultiCameraNode"]
