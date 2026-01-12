# VITA Data - 数据转换功能模块（独立实现）

此模块包含原始的数据转换功能，完全独立，不调用项目内的任何库（`gym_av_aloha`, `lerobot`）。

## 代码来源

- **原始代码**: 完全基于 `dcc5e3d` 版本的代码备份
- **独立实现**: 包含本地实现的 `ReplayBuffer` 和 `compute_stats`
- **无项目依赖**: 不调用项目内的任何库

## 模块结构

- `replay_buffer.py`: ReplayBuffer 类的独立实现（从 gym-av-aloha 复制）
- `compute_stats.py`: aggregate_stats 函数的独立实现（从 lerobot 复制）
- `datasets.py`: 数据转换函数的独立实现
- `convert.py`: 命令行转换脚本
- `__init__.py`: 模块导出

## 使用方法

### 基本用法

```bash
# 列出所有可用的数据集
python vita_data/convert.py -l

# 转换HuggingFace数据集
python vita_data/convert.py -r iantc104/av_aloha_sim_thread_needle

# 指定输出目录
python vita_data/convert.py -r iantc104/av_aloha_sim_thread_needle -o /path/to/output
```

### Python API

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from vita_data.datasets import create_av_aloha_dataset_from_lerobot

# 创建 LeRobotDataset 对象（从标准库）
datasets = [LeRobotDataset(repo_id="repo_id", episodes=list(range(0, 100)))]

# 调用转换函数（使用本地实现）
create_av_aloha_dataset_from_lerobot(
    datasets=datasets,
    root="/path/to/output",
    image_size=(240, 320),
    remove_keys=[],
)
```

## 依赖说明

### 标准库依赖（可以导入）
- `torch`, `torchvision` - PyTorch
- `numpy` - NumPy
- `zarr`, `numcodecs` - Zarr存储
- `tqdm` - 进度条
- `lerobot` - LeRobot库（用于创建LeRobotDataset对象）

### 项目内库（不导入）
- `gym_av_aloha` - 不导入，使用本地实现的 ReplayBuffer
- `lerobot` 的项目内部分 - 不导入，使用本地实现的 compute_stats

## 注意事项

- 此模块完全独立，可以单独使用
- `LeRobotDataset` 需要从标准库 `lerobot` 导入（这是外部依赖，不是项目内的库）
- 输出目录默认使用 `FLARE_DATASETS_DIR` 环境变量或 `./gym-av-aloha/outputs`
- 代码完全基于dcc5e3d版本，无任何修改
