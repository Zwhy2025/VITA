# AGENTS.md

本项目为 **VITA: Vision-to-Action Flow Matching Policy** - 用于机器人流匹配策略的开源库。

## 项目结构

- **flare/**: 主要训练和策略库 (主包)
- **lerobot/**: HuggingFace LeRobot 分支 (带有测试和 Ruff 配置)
- **gym-av-aloha/**: AV-ALOHA 仿真环境
- **gym-robomimic/**: Robomimic 任务环境

## 常用命令

### 运行训练

```bash
# 运行 VITA 训练 (Hydra 配置)
python flare/train.py policy=vita task=av_sim_close_box

# 使用自定义配置
python flare/train.py policy=vita task=av_sim_close_box device=cuda:0
```

### 运行评估

```bash
python flare/eval.py \
    checkpoint_path=outputs/ckpt/latest.pt \
    policy=vita \
    task=av_sim_close_box
```

### 测试

```bash
# 运行所有测试
make test-end-to-end

# 在 lerobot 目录下运行特定策略测试
make test-act-ete-train DEVICE=cuda
make test-diffusion-ete-eval DEVICE=cpu

# 运行单个 Python 测试文件
python -m pytest tests/test_specific.py -v

# 运行特定测试函数
python -m pytest tests/test_file.py::test_function_name -v
```

### 代码检查与格式化

```bash
# 使用 Ruff (在 lerobot/ 目录下)
cd lerobot
ruff check .
ruff check --fix .
ruff format .

# 代码安全检查
bandit -r lerobot/
```

### 包安装

```bash
# 安装主包
pip install -e .

# 安装 lerobot 包
cd lerobot && pip install -e ".[test,aloha,dev]"

# 安装所有依赖
pip install -r requirements.txt
```

## 代码风格指南

### Python 版本

- Python >= 3.10
- 使用现代类型注解 (如 `dict[str, torch.Tensor]`)

### 导入规范

1. **标准库导入** (第一组)
   ```python
   import os
   import time
   import logging
   from collections import deque
   from pathlib import Path
   from datetime import datetime
   ```

2. **第三方库导入** (第二组)
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import hydra
   import gymnasium as gym
   import numpy as np
   from omegaconf import DictConfig, OmegaConf
   from diffusers.training_utils import EMAModel
   from huggingface_hub import PyTorchModelHubMixin
   ```

3. **本地导入** (第三组)
   ```python
   from flare.factory import get_policy_class
   from flare.utils.normalize import Normalize
   from flare.policies import BasePolicy
   ```

### 命名规范

- **类名**: PascalCase (如 `VitaPolicy`, `BasePolicy`)
- **函数名**: snake_case (如 `compute_loss`, `generate_actions`)
- **私有方法**: 下划线前缀 (如 `_init_action_vae`)
- **常量**: UPPER_SNAKE_CASE
- **配置变量**: 保持与 Hydra YAML 一致 (小写+下划线)

### 类型注解

- 始终使用类型注解
- 常用类型: `dict[str, torch.Tensor]`, `tuple[torch.Tensor, None]`
- 使用 `|` 而非 `Optional` (Python 3.10+)

### 日志记录

```python
logger = logging.getLogger(__name__)

# 记录信息
logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
logger.warning("No checkpoint found, starting from scratch")
```

### 文档字符串

函数应包含简要文档字符串:

```python
def create_dataset_stats(cfg: DictConfig):
    """Create dataset metadata and statistics."""
    ...
```

### 配置管理

- 使用 Hydra 进行配置管理
- 配置文件位于 `flare/configs/`
- 使用 `DictConfig` 类型注解

### 错误处理

```python
if checkpoint_path:
    start_step = trainer.load_checkpoint(checkpoint_path)
    logger.info(f"Resumed from checkpoint at step {start_step}")
else:
    logger.warning("No checkpoint found, starting from scratch")

# 或使用 raise
raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")
```

### PyTorch 规范

- 继承 `nn.Module` 时，先调用 `super().__init__()`
- 使用 `torch.no_grad()` 装饰器进行推理
- 设备管理通过配置传递

### 格式规则

- 行长度: 110 字符 (Ruff 配置)
- 使用 4 空格缩进
- 字符串引号: 单引号优先，但保持一致
- 字典/列表末尾逗号

## 开发工作流

1. 修改代码
2. 运行相关测试: `make test-act-ete-train`
3. 运行 Ruff 检查: `ruff check . && ruff format .`
4. 验证训练脚本可以正常运行

## 注意事项

- 本项目使用 Hydra 管理配置
- 训练检查点保存于 `outputs/` 目录
- 确保环境变量 `MUJOCO_GL` 设置正确 (通常是 `egl` 或 `osmesa`)
