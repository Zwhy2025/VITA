# VITA 推理模块

基于 VITA (Vision-to-Action Flow Matching Policy) 模型的推理部署模块。

## 目录结构

```
vita_infer/
├── vita_client.py           # 客户端入口(对外暴露)
├── vita_server.py           # 服务器入口(对外暴露)
├── config/                  # 配置文件和数据（保持不变）
│   ├── global.yaml          # 全局配置文件
│   ├── scenes/              # 场景配置文件目录
│   │   ├── ur12e_real_libero_spatial.yaml
│   │   ├── dual_move_thing_to_cup_1.yaml
│   │   └── README.md
│   └── data/                # 离线测试数据目录(按场景名组织)
│       ├── ur12e_real_libero_spatial/
│       └── dual_move_thing_to_cup_1/
└── thor/                    # 核心模块
    ├── __init__.py
    ├── engine/              # 推理引擎
    │   ├── __init__.py
    │   ├── inference.py     # VitaInference 主类
    │   ├── runner.py        # VitaRunner 观测/动作管理
    │   └── loader.py        # load_vita_policy 模型加载
    ├── io/                  # 数据输入输出
    │   ├── __init__.py
    │   ├── file.py          # 离线文件数据源
    │   ├── robot.py         # 在线机器人数据源
    │   └── obs_builder.py   # 统一的观测构建逻辑
    ├── robot/               # 机器人控制
    │   ├── __init__.py
    │   ├── center.py        # InteractionDataCenter
    │   ├── nodes.py         # BaseNode, MultiArmJointNode, MultiCameraNode
    │   └── config.py        # ArmConfig, CameraConfig, RobotTopicConfig
    ├── network/             # 网络通信
    │   ├── __init__.py
    │   ├── protocol.py      # JSON 协议编解码
    │   └── client.py        # ModelClient
    └── utils/               # 工具函数
        ├── __init__.py
        ├── config.py        # load_config, merge_configs
        └── image.py         # process_image, resize_image
```

## 依赖

- Python 3.10+
- PyTorch
- 已安装的 `flare` 包（VITA 训练代码）

确保 VITA 项目已正确安装：

```bash
cd /root/workspace/VITA
pip install -e .
```

## 配置

### 配置文件结构

重构后的配置分为两部分：

1. **全局配置** (`config/global.yaml`): 包含服务器、模型、客户端基础配置
2. **场景配置** (`config/scenes/*.yaml`): 包含不同场景的模型映射配置

### 全局配置

编辑 `config/global.yaml`（只包含服务器、模型、客户端基础配置）：

```yaml
server:
  host: "0.0.0.0"
  port: 8548

model:
  checkpoint_dir: "/root/workspace/VITA/checkpoints/step_0000005000"
  mixed_precision: "bf16"  # bf16|fp16|fp32
  device: "cuda:0"  # 运行设备: cuda|cpu|cuda:0|cuda:1

client:
  server_host: "127.0.0.1"
  server_port: 8548
  send_freq: 30
  max_steps: 1000
  action_timeout: 5.0
  log_dir: "./real_logs"
```

### 场景配置

每个场景有自己的配置文件，位于 `config/scenes/` 目录下（包含场景相关的配置）：

```yaml
# 机器人硬件配置（仅在线模式使用）
datacenter:
  arms:
    right_arm:
      base_topic: "/right_arm/manip_t/controller"
      dof: 6
  cameras:
    front:
      base_topic: "/embodied/front/manip_t/sensor/camera"
      streams:
        rgb: true
        depth: false
  buffers:
    action_buffer_size: 200
    state_buffer_size: 30
    camera_buffer_size: 30
  sync:
    block_timeout: 100.0
    check_interval: 0.01
    timestamp_tolerance: 0.0015
    sync_target: "image"

# 模型输入输出映射配置
mapping:
  model_arm_order: ["right_arm"]
  model_arm_dof: 6
  use_gripper: true
  model_obs_map:
    observation.images.image: "front"
    observation.images.wrist_right_image: "wrist_right"
    observation.state: "qpos"
  image:
    normalize: true
    channel_order: "rgb"
    expected_size: [320, 240]
    resize: true
```

### 向后兼容

旧的 `config.yaml` 仍然支持，但建议迁移到新的配置结构。

## 使用方法

### 1. 启动推理服务器

```bash
cd /root/workspace/VITA/vita_infer
python vita_server.py
```

可选参数：
- `--config`: 全局配置文件路径（默认: `config/global.yaml`）
- `--log-level`: 日志级别 (DEBUG|INFO|WARNING|ERROR)

### 2. 启动客户端

在另一个终端中：

#### 在线模式（连接机器人）

```bash
# 使用场景名称（推荐）
python vita_client.py --scene ur12e_real_libero_spatial

# 或指定场景配置文件
python vita_client.py --scene-config config/scenes/ur12e_real_libero_spatial.yaml

# 或明确指定在线模式
python vita_client.py --mode online --scene ur12e_real_libero_spatial
```

#### 离线模式（从文件读取数据）

```bash
# 使用场景名（自动匹配 config/data/<scene_name> 目录）
python vita_client.py --mode offline --scene ur12e_real_libero_spatial

# 或指定离线数据目录
python vita_client.py --mode offline --offline-dir ur12e_real_libero_spatial

# 或指定完整路径
python vita_client.py --mode offline --offline-dir /path/to/data/directory
```

**参数说明：**
- `--global-config`: 全局配置文件路径（默认: `config/global.yaml`）
- `--scene`: 场景名称（对应 `config/scenes/<scene>.yaml`）
- `--scene-config`: 场景配置文件路径（覆盖 `--scene`）
- `--mode`: 运行模式 `online` 或 `offline`（如果指定了 `--offline-dir`，可省略）
- `--offline-dir`: 离线数据目录路径或子目录名（相对于 config/data 目录）。如果指定了 --scene，会自动匹配 config/data/<scene_name>

### 3. Python API 调用

```python
from thor.engine.inference import VitaInference

# 创建推理实例
infer = VitaInference(
    checkpoint_dir="/path/to/checkpoint",
    mixed_precision="bf16",
    device="cuda:0"  # 可指定具体GPU: cuda:0, cuda:1 等
)

# 构建观测
observation = {
    "observation": {
        "images": {
            "image": image_array,          # (C, H, W) float32
            "wrist_right_image": wrist_img  # (C, H, W) float32
        },
        "state": state_array  # (7,) float32
    }
}

# 更新观测并获取动作
infer.update_obs(observation)
action = infer.get_action()  # (action_horizon, action_dim)

# 重置
infer.reset()
```

## 通信协议

客户端与服务器之间使用 TCP + JSON 协议通信。

### 请求格式

```json
{
    "cmd": "infer|reset|ping",
    "obs": {...}  // 仅 infer 命令需要
}
```

### 响应格式

```json
{
    "action": [...],  // infer 命令返回
    "ok": true        // reset/ping 命令返回
}
```

## 模型参数

基于训练配置 (`ur12e_real_libero_spatial`):

| 参数 | 值 |
|------|-----|
| obs_horizon | 1 |
| action_horizon | 8 |
| pred_horizon | 16 |
| action_dim | 7 |
| state_dim | 7 |
| image_keys | `observation.images.image`, `observation.images.wrist_right_image` |
| resize_shape | (240, 320) |
| crop_shape | (224, 308) |
| fps | 30 |

## 注意事项

1. **图像预处理**: 图像需要归一化到 [0, 1]，格式为 CHW
2. **动作队列**: 模型每次生成 8 步动作，可逐步执行
3. **归一化**: 模型内部会处理状态和动作的归一化/反归一化
4. **Checkpoint**: 需要包含 `model.safetensors` 和 `config.json`

## 常见问题

### 1. 模型加载失败

确保 checkpoint 目录包含以下文件：
- `model.safetensors`: 模型权重
- `config.json`: 模型配置（由 `save_pretrained` 生成）

### 2. 图像尺寸不匹配

检查配置中的 `expected_size` 是否与实际相机分辨率匹配，或启用 `resize: true`。

### 3. 动作维度错误

确保 `mapping.model_arm_dof` 和 `use_gripper` 配置正确。
