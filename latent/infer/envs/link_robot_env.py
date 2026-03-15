"""
基于 link 库的机器人环境实现
单文件内按职责拆分为：LinkCommunicator, ObservationBuilder, check_action_safety, LinkRobotEnv
"""
import logging
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.config import ArmConfig, RobotConfig
from envs.robot_env import RobotEnvBase

logger = logging.getLogger(__name__)


# ============================================================
# 异常
# ============================================================


class ActionSafetyError(Exception):
    """动作变化超过安全阈值"""
    pass


# ============================================================
# link SDK 延迟导入
# ============================================================


def _import_link():
    """延迟导入 link 库依赖"""
    import atlas
    import link
    from link._link import SImage, Node, RawPublisher, SubscriberBase
    from manip_shared_msg.base.effector_pb2 import EffectorCommand
    from manip_shared_msg.base.gripper_pb2 import AdvancedCommand
    from manip_shared_msg.base.joint_pb2 import Joints
    from manip_shared_msg.locomotion.robot_state_pb2 import RobotState
    from manip_shared_msg.locomotion.servo_effector_pb2 import ServoEffector
    from manip_shared_msg.locomotion.servo_joint_pb2 import ServoJoint
    return {
        'atlas': atlas, 'link': link,
        'SImage': SImage, 'Node': Node,
        'RobotState': RobotState, 'ServoJoint': ServoJoint,
        'ServoEffector': ServoEffector, 'EffectorCommand': EffectorCommand,
        'AdvancedCommand': AdvancedCommand, 'Joints': Joints,
    }


# ============================================================
# 通信层
# ============================================================


class LinkCommunicator:
    """封装 link SDK 硬件通信（pub/sub、回调、指令发送）"""

    def __init__(self, arms: List[ArmConfig], cameras: Dict[str, str]):
        """
        Args:
            arms: 机械臂配置列表
            cameras: model_image_key → topic 映射
        """
        self._deps = _import_link()
        self.arms = arms
        self.cameras = cameras

        self.is_running = False
        self._node = None
        self._spin_thread = None

        # 关节状态存储
        self.joint_states: Dict[str, Dict] = {
            arm.name: {"timestamp": -1, "orig_timestamp": -1, "data": self._deps['RobotState']()}
            for arm in arms
        }
        self.arm_locks = {arm.name: threading.Lock() for arm in arms}
        self.arm_publishers: Dict[str, Any] = {}
        self.gripper_publishers: Dict[str, Any] = {}

        # 图像存储（按 model_key 索引）
        self.image_keys = list(cameras.keys())
        self.images: Dict[str, Dict] = {
            key: {"timestamp": -1, "orig_timestamp": -1, "data": None}
            for key in self.image_keys
        }
        self.image_locks = {key: threading.Lock() for key in self.image_keys}

    def start(self):
        if self.is_running:
            return

        link_mod = self._deps['link']
        link_mod.Node.Initialize("vita_robot_env")
        self._node = link_mod.GetNode()

        # 机械臂发布器 + 订阅器
        for arm in self.arms:
            self.arm_publishers[arm.name] = self._node.CreatePublisher(
                f"{arm.topic}/joint/servo", self._deps['ServoJoint'])
            self.gripper_publishers[arm.name] = self._node.CreatePublisher(
                f"{arm.topic}/gripper/servo", self._deps['ServoEffector'])
            self._node.CreateSubscriber(
                f"{arm.topic}/robot/state",
                lambda msg, name=arm.name: self._joint_cb(msg, name),
                self._deps['RobotState'])

        # 相机订阅器（直接用 model_key 作为内部标识）
        for model_key, topic in self.cameras.items():
            self._node.CreateSubscriber(
                topic,
                lambda msg, key=model_key: self._camera_cb(msg, key),
                self._deps['SImage'])

        self.is_running = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        logger.info(f"LinkCommunicator started: arms={[a.name for a in self.arms]}, "
                    f"cameras={self.image_keys}")

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)
        if self._node:
            self._node.Shutdown()
            self._node = None
        logger.info("LinkCommunicator stopped")

    def get_joint_state(self, arm_name: str) -> Tuple[List[float], float, float]:
        """返回 (joint_positions, gripper_ratio, orig_timestamp)"""
        with self.arm_locks[arm_name]:
            state = self.joint_states[arm_name]["data"]
            ts = self.joint_states[arm_name]["orig_timestamp"]
        joints = list(state.joints.position)
        eff = state.effector
        gripper = float(eff.gripper.motion.ratio) if eff.WhichOneof("status") == "gripper" else 0.0
        return joints, gripper, ts

    def get_image(self, model_key: str) -> Tuple[Optional[np.ndarray], float]:
        """返回 (rgb_array, orig_timestamp)"""
        with self.image_locks[model_key]:
            data = self.images[model_key]["data"]
            ts = self.images[model_key]["orig_timestamp"]
        return data, ts

    def send_joint_command(self, arm_name: str, joints: List[float],
                           gripper_ratio: float, speed: float = 0.5,
                           force: float = 0.5) -> None:
        ServoJoint = self._deps['ServoJoint']
        ServoEffector = self._deps['ServoEffector']
        EffectorCommand = self._deps['EffectorCommand']
        AdvancedCommand = self._deps['AdvancedCommand']
        Joints = self._deps['Joints']

        servo_joint_msg = ServoJoint()
        joints_msg = Joints()
        joints_msg.position.extend(joints)
        servo_joint_msg.target.CopyFrom(joints_msg)

        servo_gripper_msg = ServoEffector()
        effector_cmd = EffectorCommand()
        adv_cmd = AdvancedCommand()
        adv_cmd.ratio = np.clip(gripper_ratio, 0.0, 1.0)
        adv_cmd.speed = speed
        adv_cmd.force = force
        effector_cmd.adv.CopyFrom(adv_cmd)
        servo_gripper_msg.command.CopyFrom(effector_cmd)

        with self.arm_locks[arm_name]:
            self.arm_publishers[arm_name].Publish(servo_joint_msg)
            self.gripper_publishers[arm_name].Publish(servo_gripper_msg)

    # --- 内部 ---

    def _spin(self):
        while self.is_running:
            time.sleep(0.01)

    def _joint_cb(self, msg, arm_name: str):
        try:
            eff = msg.effector
            if eff.WhichOneof("status") != "gripper":
                return
            with self.arm_locks[arm_name]:
                ts = msg.header.timestamp.seconds + msg.header.timestamp.nanos * 1e-9
                self.joint_states[arm_name]["orig_timestamp"] = ts
                self.joint_states[arm_name]["timestamp"] = time.time_ns()
                self.joint_states[arm_name]["data"] = msg
        except Exception:
            logger.error(f"[{arm_name}] joint callback error", exc_info=True)

    def _camera_cb(self, msg, model_key: str):
        try:
            img_np = self._deps['atlas'].utils.image_to_numpy(msg)
            if img_np is None or img_np.size == 0:
                return
            with self.image_locks[model_key]:
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.images[model_key]["orig_timestamp"] = ts
                self.images[model_key]["timestamp"] = time.time_ns()
                self.images[model_key]["data"] = img_np
        except Exception:
            logger.error(f"[{model_key}] camera callback error", exc_info=True)


# ============================================================
# 观测构建
# ============================================================


def _process_image(rgb: Optional[np.ndarray], expected_size: Optional[Tuple[int, int]] = None,
                   normalize: bool = True, resize: bool = False) -> np.ndarray:
    """处理图像: resize → normalize → HWC→CHW"""
    if rgb is None:
        if expected_size is None:
            raise RuntimeError("expected_size required when image is missing")
        return np.zeros((3, expected_size[1], expected_size[0]), dtype=np.float32)

    if expected_size is not None:
        if rgb.shape[1] != expected_size[0] or rgb.shape[0] != expected_size[1]:
            if resize:
                import cv2
                rgb = cv2.resize(rgb, expected_size, interpolation=cv2.INTER_AREA)
            else:
                raise RuntimeError(
                    f"image size {rgb.shape[1]}x{rgb.shape[0]} != expected {expected_size[0]}x{expected_size[1]}")

    rgb = rgb.astype(np.float32)
    if normalize:
        rgb = rgb / 255.0
    return np.moveaxis(rgb, -1, 0)  # HWC → CHW


class ObservationBuilder:
    """原始传感器数据 → 模型观测格式"""

    def __init__(self, config: RobotConfig):
        self.arms = config.arms
        self.image_keys = list(config.cameras.keys())
        self.expected_size = config.image.expected_size
        self.normalize = config.image.normalize
        self.resize = config.image.resize

    def build(self, raw_images: Dict[str, np.ndarray],
              qpos: np.ndarray) -> Dict[str, Any]:
        """构建模型期望的观测

        Returns:
            {"observation": {"images": {model_key: CHW_array, ...}, "state": ndarray}}
        """
        images = {}
        for model_key in self.image_keys:
            rgb = raw_images.get(model_key)
            images[model_key] = _process_image(
                rgb, self.expected_size, self.normalize, self.resize)

        state = self._build_state(qpos)
        return {"observation": {"images": images, "state": state}}

    def _build_state(self, qpos: np.ndarray) -> np.ndarray:
        """按 arms 声明顺序截取关节状态（每臂 dof+1 含夹爪）"""
        total = sum(arm.dof + 1 for arm in self.arms)
        return qpos[:total].astype(np.float32)


# ============================================================
# 安全检查
# ============================================================


def check_action_safety(action: np.ndarray, last_action: np.ndarray,
                        arms: List[ArmConfig], threshold: float) -> None:
    """检查关节 delta 是否超过阈值，超过则 raise ActionSafetyError"""
    if len(action) != len(last_action):
        return

    joint_indices = []
    offset = 0
    for arm in arms:
        for i in range(arm.dof):
            joint_indices.append(offset + i)
        offset += arm.dof + 1  # +1 跳过夹爪

    delta = np.abs(action - last_action)
    joint_delta = delta[joint_indices]
    max_delta = np.max(joint_delta)

    if max_delta > threshold:
        idx = joint_indices[np.argmax(joint_delta)]
        raise ActionSafetyError(
            f"关节动作变化 {max_delta:.6f} 超过阈值 {threshold} (joint idx={idx})")


# ============================================================
# 环境主类
# ============================================================


class LinkRobotEnv(RobotEnvBase):
    """基于 link 库的机器人环境

    组合 LinkCommunicator（通信）+ ObservationBuilder（观测构建）+ check_action_safety（安全检查）
    """

    def __init__(self, config: RobotConfig):
        errors = config.validate()
        if errors:
            raise ValueError(f"配置验证失败: {errors}")

        self.config = config
        self.comm = LinkCommunicator(config.arms, config.cameras)
        self.obs_builder = ObservationBuilder(config)

        self.last_obs_ts: Optional[Dict] = None
        self.last_action: Optional[np.ndarray] = None

    def start(self) -> None:
        self.comm.start()

    def stop(self) -> None:
        self.comm.stop()

    def get_obs(self) -> Dict[str, Any]:
        start_time = time.time()

        while self.comm.is_running:
            if time.time() - start_time > self.config.sync.block_timeout:
                logger.warning(f"get_obs timeout ({self.config.sync.block_timeout}s)")
                return {}

            # 收集所有图像
            all_ok = True
            raw_images = {}
            obs_ts = {}
            for model_key in self.obs_builder.image_keys:
                rgb, ts = self.comm.get_image(model_key)
                if rgb is None or rgb.size == 0:
                    all_ok = False
                    break
                raw_images[model_key] = rgb
                obs_ts[f"image_ts_{model_key}"] = ts

            if not all_ok:
                time.sleep(self.config.sync.check_interval)
                continue

            # 收集关节状态
            qpos = []
            for arm in self.config.arms:
                joints, gripper, ts = self.comm.get_joint_state(arm.name)
                qpos.extend(joints)
                qpos.append(gripper)
                obs_ts[f"qpos_ts_{arm.name}"] = ts

            # 检查是否是新数据
            if not self._is_updated(obs_ts):
                time.sleep(self.config.sync.check_interval)
                continue

            self.last_obs_ts = obs_ts.copy()
            qpos_arr = np.array(qpos, dtype=np.float32)
            return self.obs_builder.build(raw_images, qpos_arr)

        return {}

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)

        # 安全检查（首帧用当前状态作为基准）
        if self.last_action is None:
            state_list = []
            for arm in self.config.arms:
                joints, gripper, _ = self.comm.get_joint_state(arm.name)
                state_list.extend(joints)
                state_list.append(gripper)
            self.last_action = np.array(state_list, dtype=np.float32)

        check_action_safety(action, self.last_action, self.config.arms,
                            self.config.action_delta_threshold)

        # 拆分动作 → 每臂 joints + gripper
        offset = 0
        for arm in self.config.arms:
            arm_len = arm.dof + 1
            if len(action) < offset + arm_len:
                break
            joints = action[offset: offset + arm.dof].tolist()
            gripper = float(action[offset + arm.dof])
            gripper = 1.0 if gripper > 0.5 else (0.0 if gripper < 0.5 else gripper)
            self.comm.send_joint_command(arm.name, joints, gripper)
            offset += arm_len

        self.last_action = action.copy()

    # --- 内部 ---

    def _is_updated(self, obs_ts: Dict) -> bool:
        if self.last_obs_ts is None:
            return True

        sync = self.config.sync
        if sync.sync_target == "image":
            for model_key in self.obs_builder.image_keys:
                key = f"image_ts_{model_key}"
                if obs_ts.get(key) != self.last_obs_ts.get(key):
                    return True
        elif sync.sync_target == "qpos":
            for arm in self.config.arms:
                key = f"qpos_ts_{arm.name}"
                prev = self.last_obs_ts.get(key)
                curr = obs_ts.get(key)
                if prev is None or curr is None or abs(curr - prev) >= sync.timestamp_tolerance:
                    return True
        return False
