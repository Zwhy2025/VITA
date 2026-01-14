# AGENTS.md

本文件包含在 VITA 仓库中工作的智能编码代理的指南和命令。

## 项目概述

VITA (Vision-to-Action Flow Matching Policy) 是一个无噪声、无条件化的视觉运动机器人策略学习框架。项目包含：

- **flare/**: 核心策略训练和实现
- **gym-av-aloha/**: AV-ALOHA 仿真环境和数据集
- **gym-robomimic/**: Robomimic 仿真环境
- **lerobot/**: LeRobot 数据集处理集成

## 构建/安装命令

### 初始设置
```bash
# 克隆并设置（推荐）
git clone git@github.com:ucd-dare/VITA.git
cd VITA
bash init.sh

# 手动设置
conda create --name vita python==3.10
conda activate vita
pip install -e .
pip install -r requirements.txt
cd lerobot && pip install -e .
cd ../gym-av-aloha && pip install -e .
cd ../gym-robomimic && pip install -e .
```

### 环境变量
```bash
export FLARE_DATASETS_DIR=<PATH_TO_VITA>/gym-av-aloha/outputs
```

## 训练命令

### 基础训练
```bash
# 在特定任务上训练 VITA 策略
python flare/train.py policy=vita task=hook_package session=test

# 覆盖特定参数
python flare/train.py policy=vita task=hook_package session=test device=cuda:2
```

### 可用任务
- AV-ALOHA: `cube_transfer`, `hook_package`, `pour_test_tube`, `slot_insertion`, `thread_needle`
- Robomimic: `robomimic_can`, `robomimic_square`
- PushT: `pusht`

### 数据集转换
```bash
# 列出可用数据集
cd gym-av-aloha/scripts
python convert.py --ls

# 转换数据集为 zarr 格式
python convert.py -r iantc104/av_aloha_sim_cube_transfer
```

## 测试命令

### 数据集测试
```bash
# 测试数据集加载和可视化
cd gym-av-aloha/scripts
python test.py
```

### LeRobot 测试（来自 lerobot/Makefile）
```bash
# 端到端测试
make test-end-to-end DEVICE=cpu

# 单个策略测试
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
make test-tdmpc-ete-train DEVICE=cpu
```

### 单个测试执行
```bash
# 运行特定测试（如果将来使用 pytest）
pytest tests/test_specific.py -v
```

## 代码风格指南

### Python 格式化
- **行长度**: 110 字符（来自 lerobot/pyproject.toml）
- **目标版本**: Python 3.10+
- **工具**: Ruff 用于代码检查和格式化

### 导入组织
```python
# 标准库导入优先
import os
import logging
from pathlib import Path

# 第三方导入
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# 本地导入
from flare.factory import get_policy_class
from flare.utils.logger import setup_logging
```

### 命名约定
- **类**: PascalCase (`VitaPolicy`, `FlowMatcher`)
- **函数/变量**: snake_case (`create_dataset`, `obs_horizon`)
- **常量**: UPPER_SNAKE_CASE (`DEFAULT_LATENT_DIM`, `MAX_STEPS`)
- **私有方法**: 前缀下划线 (`_compute_loss`)

### 类型提示
```python
from typing import Optional, Dict, Any, Tuple
import torch

def create_dataset(cfg: DictConfig, episodes: Optional[list] = None) -> Tuple[torch.utils.data.Dataset, Dict[str, Any]]:
    """创建数据集，支持可选的剧集过滤。"""
    pass
```

### 错误处理
```python
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """从文件路径加载配置。"""
    try:
        return OmegaConf.load(config_path)
    except FileNotFoundError as e:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise
```

### 配置管理
- 使用 Hydra 进行配置管理
- 默认配置在 `flare/configs/`
- 通过命令行参数覆盖
- 使用 `${}` 插值获取派生值

### 日志记录
```python
import logging

logger = logging.getLogger(__name__)

def train_step():
    logger.info("开始训练步骤")
    logger.debug(f"批次形状: {batch.shape}")
    logger.warning("使用回退配置")
    logger.error("训练失败", exc_info=True)
```

### 文档
- 为所有公共函数和类使用文档字符串
- 包含参数类型和返回值
- 为复杂函数添加使用示例

```python
def compute_flow_loss(predictions: torch.Tensor, targets: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """计算流匹配损失。
    
    Args:
        predictions: 预测流向量，形状为 (batch_size, flow_dim)
        targets: 真实流向量，形状为 (batch_size, flow_dim)
        sigma: 流匹配的噪声参数
        
    Returns:
        计算的损失张量
        
    Example:
        >>> loss = compute_flow_loss(pred, target, sigma=0.1)
    """
    pass
```

## 项目结构

### 核心组件
- `flare/policies/`: 策略实现（VITA、ACT 等）
- `flare/models/`: 神经网络架构
- `flare/flow/`: 流匹配算法
- `flare/utils/`: 训练、日志记录、检查点工具
- `flare/configs/`: Hydra 配置文件

### 数据集处理
- 数据集使用 LeRobot HuggingFace 格式
- 转换为 zarr 以加快训练速度
- 支持多种观察类型（图像、状态）

### 训练流程
- 基于 Hydra 的配置系统
- WandB 集成用于实验跟踪
- 检查点管理和恢复
- 多 GPU 训练支持

## 常见模式

### 工厂模式
```python
from flare.factory import registry

@registry.register_policy("vita")
class VitaPolicy(BasePolicy):
    pass

# 使用
policy_class = get_policy_class("vita")
```

### 配置访问
```python
@hydra.main(version_base=None)
def main(cfg: DictConfig):
    # 访问嵌套配置
    lr = cfg.optimizer_lr
    obs_horizon = cfg.policy.obs_horizon
    device = cfg.device
```

### 检查点管理
```python
from flare.utils.checkpoints import get_latest_checkpoint, save_checkpoint

# 保存检查点
save_checkpoint(model, optimizer, step, checkpoint_dir)

# 加载最新检查点
checkpoint_path = get_latest_checkpoint(checkpoint_dir)
```

## 依赖项

通过 requirements.txt 和 pyproject.toml 管理的关键依赖：
- PyTorch 生态系统（torch、torchvision）
- Hydra/OmegaConf 用于配置
- LeRobot 用于数据集处理
- WandB 用于实验跟踪
- Diffusers 用于流匹配工具
- Gymnasium 用于环境

## 代理注意事项

1. 在添加新参数前始终检查现有配置
2. 为新策略/模型遵循工厂模式
3. 在整个代码库中使用适当的日志记录
4. 在训练实验前测试数据集加载
5. 确保目标硬件的 GPU 内存使用合理
6. 使用现有检查点系统进行实验连续性