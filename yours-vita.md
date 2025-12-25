# 自定义数据集训练指南

## 概述

本文档说明如何使用修改后的 VITA 代码训练自定义数据集。代码已支持从本地 LeRobot 格式数据集进行转换和训练。

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
flare_outputs/real_libero_spatial/vita/<session>/checkpoints/
```



## 常见问题

### 1. 数据集路径错误

确保 `dataset_root` 指向转换后的数据集路径（在 `gym-av-aloha/outputs/` 下），而不是原始数据集路径。

### 2. 图像键不匹配

检查 `meta/info.json` 中的 `features` 部分，确保 `image_keys` 配置正确。

### 3. 维度不匹配

从 `info.json` 的 `features` 中确认 `state_dim` 和 `action_dim` 的正确值。

### 4. 转换失败

确保原始数据集是有效的 LeRobot 格式，包含完整的 `meta/` 目录和必要的元数据文件。

## 参考文件

- 数据集转换脚本: `gym-av-aloha/scripts/convert.py`

- 数据集加载代码: `gym-av-aloha/gym_av_aloha/datasets/av_aloha_dataset.py`

- 训练脚本: `flare/train.py`