"""
VITA 单进程推理入口
Hydra 管理配置，加载 policy + env → InferenceRunner 执行推理循环
"""
import logging

import hydra
from omegaconf import DictConfig

from policies import VitaPolicy
from envs import LinkRobotEnv
from runner import InferenceRunner
from schemas import RobotConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info('Starting inference entrypoint')

    # 初始化策略
    model_cfg = cfg.model
    logger.info(
        'Loading policy from checkpoint_dir=%s on device=%s with mixed_precision=%s',
        model_cfg.checkpoint_dir,
        model_cfg.device,
        model_cfg.mixed_precision,
    )
    policy = VitaPolicy(
        checkpoint_dir=model_cfg.checkpoint_dir,
        mixed_precision=model_cfg.mixed_precision,
        device=model_cfg.device,
    )
    policy.reset()
    logger.info('Policy initialized')

    # 初始化环境（RobotConfig 从整个 cfg 解析 link/image/sync/safety 段）
    robot_config = RobotConfig.from_dict(cfg)
    env = LinkRobotEnv(config=robot_config)
    logger.info(
        'Environment initialized with arms=%s cameras=%s',
        [arm.name for arm in robot_config.arms],
        robot_config.cameras,
    )

    # 推理循环
    runtime = cfg.runtime
    runner = InferenceRunner(
        env=env,
        policy=policy,
        max_steps=int(runtime.max_steps),
        send_freq=float(runtime.send_freq),
    )

    try:
        logger.info('Starting environment')
        env.start()
        logger.info('Environment started, entering inference loop')
        runner.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        logger.info('Stopping environment')
        env.stop()
        logger.info('Environment stopped')


if __name__ == "__main__":
    main()
