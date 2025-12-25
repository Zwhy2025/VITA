# VITA 推理模块

基于 VITA (Vision-to-Action Flow Matching Policy) 模型的推理部署模块。

## 目录结构

```
vita_infer/
├── vita_inference.py   # 核心推理类
├── vita_server.py      # TCP 推理服务器
├── vita_client.py      # 客户端(连接 datacenter)
├── config.yaml         # 配置文件
├── protocol.py         # 网络通信协议
├── datacenter.py       # 机器人数据中心
└── README.md           # 本文档
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

编辑 `config.yaml` 配置文件：

### 服务器配置

```yaml
server:
  host: "0.0.0.0"
  port: 8548
```

### 模型配置

```yaml
model:
  checkpoint_dir: "/root/workspace/VITA/checkpoints/step_0000005000"
  mixed_precision: "bf16"  # bf16|fp16|fp32
  device: "cuda"  # 运行设备: cuda|cpu|cuda:0|cuda:1 (可指定具体GPU)
```

### 机器人配置

根据实际机器人配置修改 `datacenter` 和 `mapping` 部分。

## 使用方法

### 1. 启动推理服务器

```bash
cd /root/workspace/DRRM/vita_infer
python vita_server.py --config config.yaml
```

可选参数：
- `--config`: 配置文件路径（默认: `config.yaml`）
- `--log-level`: 日志级别 (DEBUG|INFO|WARNING|ERROR)

### 2. 启动客户端

在另一个终端中：

```bash
python vita_client.py --config config.yaml
```

### 3. Python API 调用

```python
from vita_inference import VitaInference

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

