# 自定义数据集训练和部署指南

## 目录

- [概述](#概述)
- [主要改动](#主要改动)
- [步骤 1: 准备数据集](#步骤-1-准备数据集)
- [步骤 2: 转换数据集](#步骤-2-转换数据集)
- [步骤 3: 创建任务配置文件](#步骤-3-创建任务配置文件)
- [步骤 4: 设置环境变量](#步骤-4-设置环境变量)
- [步骤 5: 开始训练](#步骤-5-开始训练)
- [步骤 6: 监控训练](#步骤-6-监控训练)
- [步骤 7: 模型部署](#步骤-7-模型部署)
- [步骤 8: 实际示例](#步骤-8-实际示例)
- [常见问题](#常见问题)
- [训练配置参考](#训练配置参考)
- [参考文件](#参考文件)
- [工作流程总结](#工作流程总结)
- [附录：检查点目录结构](#附录检查点目录结构)

## 概述

本文档说明如何使用修改后的 VITA 代码训练自定义数据集并部署模型。代码已支持从本地 LeRobot 格式数据集进行转换、训练和推理部署。

## 主要改动

根据 git diff，主要改动包括：

1. **`gym-av-aloha/gym_av_aloha/datasets/av_aloha_dataset.py`**

- 添加 `dataset_root` 参数，支持从本地路径加载数据集
- 区分输入数据集路径和输出数据集路径

2. **`gym-av-aloha/scripts/convert.py`**

- `convert_dataset` 函数现在可以接受本地路径或 HuggingFace repo_id
- 自动检测本地路径并加载元数据
- 为本地数据集生成默认配置

## 步骤 1: 准备数据集

### 数据集格式要求

你的数据集必须是 **LeRobot 格式**，包含以下目录结构：

```javascript
your_dataset/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── episodes.jsonl     # episode 信息
│   ├── episodes_stats.jsonl
│   ├── stats.json
│   └── tasks.jsonl
├── data/                  # 数据文件（parquet 格式）
├── images/                # 图像文件（可选）
└── videos/                # 视频文件
```



### 数据集元信息检查

确保 `meta/info.json` 包含：

- `total_episodes`: episode 总数
- `fps`: 帧率
- `features`: 数据特征定义（包括图像键、状态键、动作键等）

## 步骤 2: 转换数据集

### 转换本地数据集

使用修改后的转换脚本将本地数据集转换为训练格式：

```bash
cd gym-av-aloha/scripts
python convert.py -r /path/to/your/dataset
```

例如，对于你的数据集：

```bash
python convert.py -r /root/workspace/VITA/datasets/ur12e/real_libero_spatial
```



### 转换说明

- 脚本会自动检测路径是否为本地目录

- 自动从 `meta/info.json` 读取数据集信息

- 生成 `repo_id` 为路径的最后两级（如 `ur12e/real_libero_spatial`）

- 默认配置：
- 处理所有 episodes

- 不删除任何键

- 保持原始图像尺寸

### 转换输出

转换后的数据集保存在：

```javascript
gym-av-aloha/outputs/<repo_id>/
├── config.json           # 数据集配置
├── data/                 # zarr 格式数据
└── meta/                 # 元数据
```



## 步骤 3: 创建任务配置文件

在 `flare/configs/task/` 目录下创建你的任务配置文件，例如 `real_libero_spatial.yaml`：

```yaml
# @package _global_

resize_shape: [480, 640]  # 根据你的图像尺寸调整
crop_shape: [224, 308]    # 根据你的需求调整

task:
  name: real_libero_spatial
  dataset_repo_id: ur12e/real_libero_spatial  # 与转换时生成的 repo_id 一致
  dataset_root: ${oc.env:FLARE_DATASETS_DIR}/ur12e/real_libero_spatial
  dataset_episodes: null  # null 表示使用所有 episodes
  
  # 图像归一化统计（可选，如果不设置会使用数据集统计）
  override_stats:
    observation.images.image:  # 根据你的图像键名称调整
      mean: [[[0.485]], [[0.456]], [[0.406]]]
      std: [[[0.229]], [[0.224]], [[0.225]]]

  # 环境配置（如果不需要在线评估可以省略）
  env_package: null
  env_name: null
  env_kwargs: null

  fps: 30  # 从 info.json 中获取
  image_keys:
    - observation.images.image  # 根据你的数据集调整
    # - observation.images.wrist_right_image  # 如果有多个图像
  state_key: observation.state
  action_key: action
  state_dim: 7  # 根据你的数据集调整
  action_dim: 7  # 根据你的数据集调整
```



### 配置参数说明

- `dataset_repo_id`: 必须与转换时生成的 repo_id 一致

- `dataset_root`: 指向转换后的数据集路径（在 `gym-av-aloha/outputs/` 下）

- `fps`: 从原始数据集的 `meta/info.json` 中获取

- `image_keys`: 列出所有要使用的图像键

- `state_dim` / `action_dim`: 从 `info.json` 的 features 中获取

## 步骤 4: 设置环境变量

确保设置了数据集目录环境变量：

```bash
export FLARE_DATASETS_DIR=/root/workspace/VITA/gym-av-aloha/outputs
```



或者添加到 `~/.bashrc`：

```bash
echo 'export FLARE_DATASETS_DIR=/root/workspace/VITA/gym-av-aloha/outputs' >> ~/.bashrc
source ~/.bashrc
```



## 步骤 5: 开始训练

### 基本训练命令

```bash
python flare/train.py policy=vita task=real_libero_spatial session=my_training_session
```



### 常用训练选项

```bash
# 指定 GPU
python flare/train.py policy=vita task=real_libero_spatial session=test device=cuda:0

# 调整验证频率
python flare/train.py policy=vita task=real_libero_spatial session=test \
  val.val_offline_freq=1000 val.num_episodes=5

# 调整训练步数和批次大小
python flare/train.py policy=vita task=real_libero_spatial session=test \
  train.steps=50000 train.batch_size=64

# 禁用在线评估（如果没有环境）
python flare/train.py policy=vita task=real_libero_spatial session=test \
  val.val_online_freq=0
```



## 步骤 6: 监控训练

### WandB 配置

训练使用 WandB 进行日志记录。在 `flare/configs/default_policy.yaml` 中配置：

```yaml
wandb:
  enable: true
  project: vita-real-libero-spatial
  entity: "your_wandb_entity"  # 设置你的 WandB entity
```



或通过命令行参数：

```bash
python flare/train.py policy=vita task=real_libero_spatial session=test \
  wandb.entity=your_entity wandb.project=my_project
```



### 检查点

训练检查点保存在：

```javascript
flare_outputs/<task_name>/vita/<session>/checkpoints/
├── step_0000001000/          # 按步数保存的检查点
│   ├── model.safetensors      # 模型权重
│   ├── config.json            # 模型配置
│   └── training_state.pt      # 训练状态（优化器、调度器等）
├── step_0000002000/
└── ...
```

检查点命名规则：`step_<步数>`（10位数字，前导零填充）

#### 检查点保存频率

在 `flare/configs/default_policy.yaml` 中配置：

```yaml
train:
  save_freq: 1000      # 每 N 步保存一次检查点
  keep_freq: 20000     # 保留每 N 步的检查点（删除更早的）
```

#### 查找最新检查点

```bash
# 方法1: 手动查找
ls -t flare_outputs/<task_name>/vita/<session>/checkpoints/ | head -1

# 方法2: 使用 Python 脚本
python -c "from flare.utils.checkpoints import get_latest_checkpoint; print(get_latest_checkpoint('flare_outputs/<task_name>/vita/<session>/checkpoints'))"
```

#### 恢复训练

```bash
# 从最新检查点恢复
python flare/train.py policy=vita task=real_libero_spatial session=test resume=true

# 从指定检查点恢复
python flare/train.py policy=vita task=real_libero_spatial session=test \
  resume=true checkpoint_path=flare_outputs/<task_name>/vita/<session>/checkpoints/step_0000005000
```

---

## 步骤 7: 模型部署

训练完成后，需要将模型部署为推理服务。VITA 提供了完整的推理模块 `vita_infer`。

### 7.1 准备检查点

确保检查点目录包含以下文件：

```javascript
checkpoint_dir/
├── model.safetensors      # 必需：模型权重
├── config.json            # 必需：模型配置
└── training_state.pt      # 可选：训练状态（推理不需要）
```

推理模块会自动从检查点目录的父目录查找训练配置文件 `train_config_*.yaml`（位于 `logs/` 目录）。

### 7.2 配置推理服务器

编辑 `vita_infer/config.yaml`：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"        # 监听地址
  port: 8548              # 监听端口

# 模型配置
model:
  checkpoint_dir: "/root/workspace/VITA/flare_outputs/<task_name>/vita/<session>/checkpoints/step_0000005000"
  mixed_precision: "bf16"  # 计算精度: bf16|fp16|fp32
  device: "cuda:0"         # 运行设备: cuda|cpu|cuda:0|cuda:1

# 客户端配置（如果使用客户端）
client:
  server_host: "127.0.0.1"
  server_port: 8548
  send_freq: 30            # 动作发送频率 (Hz)，与训练 fps 保持一致
  max_steps: 1000
  action_timeout: 5.0
  log_dir: "./real_logs"

# 机器人硬件配置（根据实际情况修改）
datacenter:
  arms:
    right_arm:
      base_topic: "/right_arm/manip_t/controller"
      dof: 6

  cameras:
    wrist_right:
      base_topic: "/right_arm/manip_t/sensor/camera"
      streams:
        rgb: true
        depth: false
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
  # 机械臂配置：根据训练时的配置调整
  model_arm_order: ["right_arm"]
  model_arm_dof: 6
  use_gripper: true
  fill_missing_arm: false

  # 观测映射：模型期望键名 -> datacenter 数据源
  model_obs_map:
    observation.images.image: "front"              # 主相机图像
    observation.images.wrist_right_image: "wrist_right"  # 手腕图像
    observation.state: "qpos"                      # 机器人状态

  # 图像预处理配置
  image:
    normalize: true              # 归一化到 [0, 1]
    channel_order: "rgb"        # 通道顺序: rgb|bgr
    expected_size: [320, 240]   # 期望图像尺寸: [宽, 高]
    resize: true                # 尺寸不匹配时是否缩放
```

**重要配置说明：**

- `checkpoint_dir`: 指向训练好的检查点目录
- `device`: 指定使用的 GPU（如 `cuda:0`、`cuda:1`）
- `model_obs_map`: 必须与训练时的 `image_keys` 和 `state_key` 一致
- `expected_size`: 必须与训练时的 `resize_shape` 一致（注意顺序是 [宽, 高]）

### 7.3 启动推理服务器

```bash
cd /root/workspace/VITA/vita_infer
python vita_server.py --config config.yaml
```

可选参数：
- `--config`: 配置文件路径（默认: `config.yaml`）
- `--log-level`: 日志级别 (DEBUG|INFO|WARNING|ERROR)

服务器启动后会：
1. 加载模型检查点
2. 加载训练配置（从 `logs/train_config_*.yaml`）
3. 加载数据集统计信息（用于归一化）
4. 监听指定端口，等待客户端连接

### 7.4 使用 Python API 直接调用

如果不使用服务器模式，可以直接使用 Python API：

```python
from vita_inference import VitaInference

# 创建推理实例
infer = VitaInference(
    checkpoint_dir="/path/to/checkpoint",
    mixed_precision="bf16",
    device="cuda:0"
)

# 构建观测
observation = {
    "observation": {
        "images": {
            "image": image_array,              # (C, H, W) float32, [0, 1]
            "wrist_right_image": wrist_img     # (C, H, W) float32, [0, 1]
        },
        "state": state_array                   # (state_dim,) float32
    }
}

# 更新观测并获取动作
infer.update_obs(observation)
action = infer.get_action()  # (action_horizon, action_dim)

# 重置（开始新的 episode）
infer.reset()
```

### 7.5 启动客户端（连接机器人）

如果使用机器人硬件，启动客户端：

```bash
cd /root/workspace/VITA/vita_infer
python vita_client.py --config config.yaml
```

客户端会：
1. 连接到推理服务器
2. 从机器人硬件（datacenter）获取观测数据
3. 发送到服务器进行推理
4. 接收动作并发送给机器人执行

### 7.6 离线推理测试

使用离线数据测试推理功能：

```bash
cd /root/workspace/VITA/vita_infer
python vita_client_offline.py --config config.yaml --data-dir ./offline
```

离线数据格式：

```javascript
offline/
├── qpos.csv                    # 机器人状态（CSV格式，第一行为表头）
├── <camera_name>;color.png     # 相机图像（如 front;color.png）
└── ...
```

---

## 步骤 8: 实际示例

### 示例 1: 单臂任务（ur12e_real_libero_spatial）

**数据集转换：**
```bash
cd gym-av-aloha/scripts
python convert.py -r /root/workspace/VITA/datasets/ur12e/real_libero_spatial
```

**任务配置：** `flare/configs/task/ur12e_real_libero_spatial.yaml`
- `state_dim: 7`（6 dof + 1 gripper）
- `action_dim: 7`
- `image_keys`: `observation.images.image`, `observation.images.wrist_right_image`

**训练：**
```bash
python flare/train.py policy=vita task=ur12e_real_libero_spatial session=test device=cuda:0
```

**部署配置：** `vita_infer/config.yaml`
- `model_arm_dof: 6`
- `use_gripper: true`
- `model_obs_map` 包含两个图像键

### 示例 2: 双臂任务（dual_move_thing_to_cup_1）

**数据集转换：**
```bash
cd gym-av-aloha/scripts
python convert.py -r /root/workspace/VITA/datasets/r2v2/dual_move_thing_to_cup_1
```

**任务配置：** `flare/configs/task/dual_move_thing_to_cup_1.yaml`
- `state_dim: 16`（双臂：每臂 7 维）
- `action_dim: 16`
- `image_keys`: `observation.images.image`, `observation.images.wrist_right_image`, `observation.images.wrist_left_image`

**训练：**
```bash
python flare/train.py policy=vita task=dual_move_thing_to_cup_1 session=test device=cuda:0
```

**部署配置：** 需要修改 `mapping` 部分以支持双臂

---

## 常见问题

### 1. 数据集路径错误

确保 `dataset_root` 指向转换后的数据集路径（在 `gym-av-aloha/outputs/` 下），而不是原始数据集路径。

### 2. 图像键不匹配

检查 `meta/info.json` 中的 `features` 部分，确保 `image_keys` 配置正确。

### 3. 维度不匹配

从 `info.json` 的 `features` 中确认 `state_dim` 和 `action_dim` 的正确值。

### 4. 转换失败

确保原始数据集是有效的 LeRobot 格式，包含完整的 `meta/` 目录和必要的元数据文件。

### 5. 检查点加载失败

**问题：** 推理时找不到训练配置文件

**解决：** 确保检查点目录结构如下：
```javascript
checkpoints/
└── step_0000005000/
    ├── model.safetensors
    └── config.json

logs/
└── train_config_202512252356.yaml  # 推理模块会查找此文件
```

如果配置文件不在 `logs/` 目录，可以手动指定：
```python
from vita_inference import load_vita_policy

model = load_vita_policy(
    checkpoint_dir="/path/to/checkpoint",
    config_yaml_path="/path/to/train_config.yaml"
)
```

### 6. 推理时图像尺寸不匹配

**问题：** `expected_size` 配置错误

**解决：** 
- 检查训练配置中的 `resize_shape`（格式：`[H, W]`）
- 推理配置中的 `expected_size` 格式为 `[W, H]`（注意顺序相反）
- 例如：训练时 `resize_shape: [240, 320]`，推理时 `expected_size: [320, 240]`

### 7. 动作维度不匹配

**问题：** 模型输出的动作维度与机器人不匹配

**解决：**
- 检查 `mapping.model_arm_dof` 和 `use_gripper` 配置
- 确保与训练时的 `action_dim` 一致
- 单臂：`model_arm_dof: 6` + `use_gripper: true` = 7 维
- 双臂：需要根据实际情况调整

### 8. 归一化统计信息缺失

**问题：** 推理时找不到数据集的归一化统计信息

**解决：**
- 推理模块会尝试从多个路径查找 `meta/stats.json`
- 确保原始数据集路径正确，或手动设置数据集路径
- 如果使用 `override_stats`，确保配置正确

### 9. 服务器连接失败

**问题：** 客户端无法连接到推理服务器

**解决：**
- 检查服务器是否正常启动（查看日志）
- 检查防火墙设置
- 确认 `server_host` 和 `server_port` 配置正确
- 测试连接：`telnet <server_host> <server_port>`

### 10. 推理速度慢

**优化建议：**
- 使用 `mixed_precision: "bf16"` 或 `"fp16"`（需要 GPU 支持）
- 确保使用 GPU：`device: "cuda:0"`
- 减少图像尺寸（如果可能）
- 检查是否有其他进程占用 GPU

### 11. 内存不足

**问题：** GPU 内存不足

**解决：**
- 使用更小的批次大小（如果训练时）
- 使用 `mixed_precision: "bf16"` 减少内存占用
- 关闭其他占用 GPU 的程序
- 使用多 GPU：`device: "cuda:1"`（如果有多块 GPU）

---

## 训练配置参考

### 常用训练参数

```yaml
# 训练步数和批次大小
train:
  steps: 30000          # 总训练步数
  batch_size: 512       # 批次大小（根据 GPU 内存调整）
  num_workers: 4        # 数据加载线程数
  save_freq: 1000       # 检查点保存频率
  keep_freq: 20000      # 保留检查点的频率

# 优化器配置
optimizer_lr: 1e-4              # 学习率
optimizer_lr_backbone: 1e-5     # 骨干网络学习率
optimizer_betas: [0.95, 0.999]
optimizer_weight_decay: 1e-6

# 验证配置
val:
  num_episodes: 0        # 验证集 episode 数（0 表示不使用验证集）
  val_offline_freq: 0    # 离线验证频率（0 表示不进行离线验证）
  val_online_freq: 0     # 在线验证频率（0 表示不进行在线验证）
```

### 图像预处理配置

```yaml
# 在任务配置文件中
resize_shape: [240, 320]   # 图像缩放尺寸 [H, W]
crop_shape: [224, 308]     # 图像裁剪尺寸 [H, W]

# 图像归一化（可选）
override_stats:
  observation.images.image:
    mean: [[[0.485]], [[0.456]], [[0.406]]]  # ImageNet 均值
    std: [[[0.229]], [[0.224]], [[0.225]]]   # ImageNet 标准差
```

---

## 参考文件

### 核心代码文件

- **数据集转换脚本**: `gym-av-aloha/scripts/convert.py`
- **数据集加载代码**: `gym-av-aloha/gym_av_aloha/datasets/av_aloha_dataset.py`
- **训练脚本**: `flare/train.py`
- **策略实现**: `flare/policies/vita/vita_policy.py`
- **训练器**: `flare/trainers/policy_trainer.py`

### 配置文件

- **默认训练配置**: `flare/configs/default_policy.yaml`
- **默认网络配置**: `flare/configs/default_network.yaml`
- **VITA 策略配置**: `flare/configs/policy/vita.yaml`
- **任务配置**: `flare/configs/task/*.yaml`

### 推理模块

- **推理核心**: `vita_infer/vita_inference.py`
- **推理服务器**: `vita_infer/vita_server.py`
- **客户端（机器人）**: `vita_infer/vita_client.py`
- **离线客户端**: `vita_infer/vita_client_offline.py`
- **数据中心**: `vita_infer/datacenter.py`
- **通信协议**: `vita_infer/protocol.py`
- **推理配置**: `vita_infer/config.yaml`

### 工具函数

- **检查点管理**: `flare/utils/checkpoints.py`
- **日志记录**: `flare/utils/logger.py`

---

## 工作流程总结

完整的训练和部署流程：

1. **准备数据集** → LeRobot 格式数据集
2. **转换数据集** → `convert.py` 转换为 zarr 格式
3. **创建任务配置** → `flare/configs/task/<task_name>.yaml`
4. **设置环境变量** → `FLARE_DATASETS_DIR`
5. **开始训练** → `python flare/train.py`
6. **监控训练** → WandB 或日志文件
7. **选择检查点** → 从 `checkpoints/` 目录选择最佳模型
8. **配置推理** → 编辑 `vita_infer/config.yaml`
9. **启动服务器** → `python vita_server.py`
10. **测试部署** → 使用离线数据或连接机器人

---

## 附录：检查点目录结构

完整的检查点目录结构示例：

```javascript
flare_outputs/
└── dual_move_thing_to_cup_1/
    └── vita/
        └── t3/
            ├── checkpoints/
            │   ├── step_0000001000/
            │   │   ├── model.safetensors
            │   │   ├── config.json
            │   │   └── training_state.pt
            │   ├── step_0000002000/
            │   └── ...
            ├── logs/
            │   ├── train.log
            │   └── train_config_202512252356.yaml  # 推理时需要
            └── eval/
                └── ...
```

推理时需要的文件：
- `checkpoints/step_XXXXX/model.safetensors` - 模型权重
- `checkpoints/step_XXXXX/config.json` - 模型配置
- `logs/train_config_*.yaml` - 训练配置（包含数据集路径、归一化统计等）