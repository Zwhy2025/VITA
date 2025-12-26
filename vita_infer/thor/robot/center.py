"""
机器人交互中心
组合关节/相机节点，提供统一的观测和动作接口
"""
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from thor.robot.config import RobotTopicConfig
from thor.robot.nodes import MultiArmJointNode, MultiCameraNode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("robot_interaction")


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
            # 默认从 config/global.yaml 加载，但需要提取 datacenter 部分
            config_path = Path(__file__).parent.parent.parent / "config" / "global.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"默认配置文件不存在: {config_path}")
            
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

                success = self.arm_node.send_servoj(arm.name, joints, gripper)
                all_success = all_success and success

                offset += arm_len

            return all_success

        except Exception as e:
            self.logger.error(f"动作发布失败：{e}")
            return False


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


__all__ = ["InteractionDataCenter", "test_spin"]

