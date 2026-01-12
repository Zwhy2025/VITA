# 场景配置文件说明

## 概述

场景配置文件用于定义不同场景下的模型输入输出映射配置。这些配置会与全局配置文件（`config/global.yaml`）合并。

## 配置文件结构

每个场景配置文件应包含 `mapping` 部分，定义：
- 机械臂配置（`model_arm_order`, `model_arm_dof`, `use_gripper`, `fill_missing_arm`）
- 观测映射（`model_obs_map`）：模型期望的键名 -> 数据源
- 图像预处理配置（`image`）

## 使用方式

### 离线客户端

```bash
# 使用场景名称（自动匹配 config/data/<scene_name> 目录和 config/scenes/<scene_name>.yaml 配置）
python vita_client.py --mode offline --scene ur12e_real_libero_spatial

# 或指定离线数据目录
python vita_client.py --mode offline --offline-dir ur12e_real_libero_spatial

# 或指定场景配置文件路径和数据目录
python vita_client.py --mode offline --scene-config config/scenes/ur12e_real_libero_spatial.yaml --offline-dir ur12e_real_libero_spatial
```

### 在线客户端

```bash
# 使用场景名称
python vita_client.py --scene ur12e_real_libero_spatial

# 或指定场景配置文件路径
python vita_client.py --scene-config config/scenes/ur12e_real_libero_spatial.yaml
```

## 创建新场景配置

1. 在 `config/scenes/` 目录下创建新的 YAML 文件，例如 `my_scene.yaml`
2. 在 `config/data/` 目录下创建对应的数据目录，例如 `config/data/my_scene/`
3. 参考现有配置文件的结构
4. 根据你的场景修改 `mapping` 部分

**注意**：场景配置文件名和数据目录名应该保持一致，这样使用 `--scene` 参数时可以自动匹配。

## 配置合并规则

- 场景配置会覆盖全局配置中同名的键
- 对于字典类型的配置（如 `mapping`），会进行深度合并
- 如果场景配置中缺少某些键，会使用全局配置的默认值

