# VITA 推理模块

基于 VITA (Vision-to-Action Flow Matching Policy) 模型的单进程推理部署。

## 目录结构

```
latent/infer/
├── infer.py                # Hydra 入口
├── schemas.py              # RobotConfig, ArmConfig, ImageConfig, SyncConfig
├── runner.py               # InferenceRunner 推理编排
├── policies/               # 策略层
│   ├── base_policy.py      # PolicyBase 抽象基类
│   ├── vita_policy.py      # VitaPolicy 实现
│   └── loader.py           # 模型加载（config + weights + stats）
├── envs/                   # 环境层
│   ├── robot_env.py        # RobotEnvBase 抽象基类
│   ├── link_robot_env.py   # LinkRobotEnv + LinkCommunicator + ObservationBuilder
└── configs/                # Hydra 配置
    ├── default.yaml        # 默认配置（模型 + 运行时）
    └── robot/              # 按机器人类型分
        ├── base.yaml       # 机器人共享默认值
        ├── ur12e.yaml      # UR12e 单臂 6-DOF
        └── dual_r2v2.yaml  # R2V2 双臂 7-DOF
```

## 架构

```
infer.py (Hydra 入口)
  └── InferenceRunner (编排)
        ├── PolicyBase → VitaPolicy (策略推理)
        └── RobotEnvBase → LinkRobotEnv (硬件交互)
                              ├── LinkCommunicator (link SDK 通信)
                              ├── ObservationBuilder (观测构建)
                              └── check_action_safety (安全检查)
```

- **PolicyBase / VitaPolicy**: 策略抽象，`predict(obs) → action`
- **RobotEnvBase / LinkRobotEnv**: 环境抽象，`get_obs() → obs`，`step(action)`
- **InferenceRunner**: 编排 env + policy 的推理主循环

## 使用

```bash
# 默认机器人（ur12e）
python infer.py

# 指定机器人类型
python infer.py robot=dual_r2v2

# 覆盖参数
python infer.py model.device=cuda:1 runtime.max_steps=500
```

## 配置

使用 Hydra 管理配置，按机器人类型组织。机械臂使用 `base_topic` 表示控制路径前缀，
相机配置仍直接映射模型 image key 到完整订阅 topic：

```yaml
# configs/robot/ur12e.yaml
defaults:
  - base
  - _self_

link:
  cameras:
    # model image key → topic（直接映射，无中间层）
    image: "/embodied/front/manip_t/sensor/camera/color"
    wrist_right_image: "/right_arm/manip_t/sensor/camera/color"
  arms:
    # 声明顺序 = 模型 state/action 向量中的臂顺序
    right_arm:
      base_topic: "/right_arm/manip_t/controller"
      dof: 6    # 必须配置

safety:
  action_delta_threshold: 0.2
```

## Python API

```python
from policies.vita_policy import VitaPolicy
from envs.link_robot_env import LinkRobotEnv
from schemas import RobotConfig

policy = VitaPolicy(checkpoint_dir="/path/to/checkpoint")
config = RobotConfig.from_dict(cfg)
env = LinkRobotEnv(config=config)

env.start()
obs = env.get_obs()
action = policy.predict(obs)  # (action_horizon, action_dim)
env.step(action[0])
env.stop()
```

## 依赖

- Python 3.10+, PyTorch, Hydra
- `flare` 包（VITA 训练代码）
- `link` + `atlas`（机器人 SDK，仅 LinkRobotEnv 需要）

## 注意事项

1. **图像格式**: CHW, float32, 归一化到 [0, 1]
2. **动作安全**: 相邻帧关节变化超过 `action_delta_threshold` 会触发 `ActionSafetyError` 紧急停止
3. **Checkpoint**: 需包含 `model.safetensors` + 训练 config YAML

## 未来重构方向

### 模型加载独立化
当前推理直接依赖 `flare` 训练代码（通过 `flare.factory.get_policy_class()` 动态实例化训练侧模型）。理想方案是将 config + stats 嵌入 safetensors checkpoint header，导出 inference-only 模型，但需要训练侧配合改 checkpoint 导出逻辑。

### LinkRobotEnv 进一步解耦
当前 `LinkCommunicator` 仍然直接构造 protobuf 消息。如果需要支持更多机器人 SDK，可以进一步抽象通信协议层。
