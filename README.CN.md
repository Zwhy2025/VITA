<p align="center">
  <img src=".assets/vita.png" alt="VITA" />
</p>

# 🌊 VITA: 视觉到动作的流匹配策略

本仓库提供了论文 **VITA: Vision-to-Action Flow Matching Policy**（2025年7月）的官方实现。

**VITA** 是一个**无噪声、无条件**的策略学习框架，通过直接将潜在图像映射到潜在动作来学习视觉运动策略。

<p align="center">
  <a href="https://ucd-dare.github.io/VITA/"><img src="https://img.shields.io/badge/Project%20Page-%F0%9F%94%8D-blue" alt="项目页面"></a>
  <a href="https://arxiv.org/abs/2507.13231"><img src="https://img.shields.io/badge/arXiv-2507.13231-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2507.13231"><img src="https://img.shields.io/badge/PDF-%F0%9F%93%84-blue" alt="PDF"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="许可证">
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/ucd-dare/VITA/refs/heads/gh-pages/static/videos/vita_anim.gif" width="100%" />
</p>

---

> \[!注意\]
> - **2025年12月** ArXiv 更新至 V3 版本，包含多个新的真实世界任务和更多讨论。
> - **2025年11月：** 我们已将 `VITA` 和 Diffusion Transformer 实现集成到 [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse) [PR#580](https://github.com/RoboVerseOrg/RoboVerse/pull/580)。
> - **2025年10月：** 代码已发布。ArXiv 更新至 V2。
> - **2025年7月** 论文发布在 ArXiv。

---

## 🚀 快速开始

本节涵盖安装、数据集预处理和训练。

* **策略和训练：** `./flare`
* **仿真：** [AV-ALOHA](https://soltanilara.github.io/av-aloha/) 任务（`gym-av-aloha`）和 [Robomimic](https://robomimic.github.io/) 任务（`gym-robomimic`）
* **数据集：** 基于 [LeRobot](https://github.com/huggingface/lerobot) Hugging Face 格式构建，并优化预处理为离线 Zarr 格式以加快训练速度

---

### 🔧 安装

#### 方法一：使用自动化安装脚本（推荐）

```bash
git clone git@github.com:ucd-dare/VITA.git
cd VITA
bash init.sh
```

脚本会自动完成以下操作：
- 创建并激活 conda 环境 `vita`
- 安装所有必要的依赖
- 设置环境变量 `FLARE_DATASETS_DIR`

#### 方法二：手动安装

```bash
git clone git@github.com:ucd-dare/VITA.git
cd VITA
conda create --name vita python==3.10
conda activate vita
conda install cmake
pip install -e .
pip install -r requirements.txt
# 安装 LeRobot 依赖
cd lerobot
pip install -e .
# 安装 ffmpeg 用于数据集处理
conda install -c conda-forge ffmpeg
```

设置数据集存储路径：

```bash
echo 'export FLARE_DATASETS_DIR=<PATH_TO_VITA>/gym-av-aloha/outputs' >> ~/.bashrc
# 重新加载 bashrc
source ~/.bashrc
conda activate vita
```

根据需要安装 AV-ALOHA 和/或 Robomimic 的基准测试依赖：

* **AV-ALOHA**

```bash
cd gym-av-aloha
pip install -e .
```

* **Robomimic**

```bash
cd gym-robomimic
pip install -e .
```

---

### 📦 数据集预处理

我们的数据加载器扩展了 [LeRobot](https://github.com/huggingface/lerobot)，将数据集转换为离线 zarr 格式以加快训练速度。我们在 HuggingFace 上托管数据集。要列出可用数据集：

```bash
cd gym-av-aloha/scripts
python convert.py --ls
```

截至 2025 年 9 月，可用数据集包括：

```yaml
- iantc104/av_aloha_sim_cube_transfer
- iantc104/av_aloha_sim_thread_needle
- iantc104/av_aloha_sim_pour_test_tube
- iantc104/av_aloha_sim_slot_insertion
- iantc104/av_aloha_sim_hook_package
- iantc104/robomimic_sim_square
- iantc104/robomimic_sim_can
- lerobot/pusht
```

将 HuggingFace 数据集（转换可能需要 >10 分钟）转换为离线 zarr 数据集。例如：

```bash
# 替换数据集标志以使用其他任务...

# AV-ALOHA
python convert.py -r iantc104/av_aloha_sim_thread_needle
python convert.py -r iantc104/av_aloha_sim_cube_transfer
python convert.py -r iantc104/av_aloha_sim_hook_package
...

# Robomimic
python convert.py -r iantc104/robomimic_sim_square
python convert.py -r iantc104/robomimic_sim_can
...
```

数据集将存储在 `./gym-av-aloha/outputs`。

如果在转换过程中遇到 `cv2`、`numpy` 或 `scipy` 的错误，重新安装它们通常可以解决问题：

```bash
pip uninstall opencv-python numpy scipy
pip install opencv-python numpy scipy
```

**numpy/pandas 兼容性错误**：如果遇到 `ValueError: numpy.dtype size changed` 错误，这通常是由于 numpy 和 pandas 版本不兼容导致的。解决方法：

```bash
pip install --upgrade --force-reinstall numpy pandas
```

---

### 📊 日志记录

我们使用 [WandB](https://wandb.ai/) 进行实验跟踪。使用 `wandb login` 登录，然后在 `./flare/configs/default_policy.yaml` 中设置您的实体（或在训练命令后追加 `wandb.entity=YOUR_ENTITY_NAME`）：

```yaml
wandb:
  entity: "YOUR_WANDB_ENTITY"
```

我们记录：离线验证结果、在线仿真器验证结果，以及 ODE 去噪过程的可视化，这有助于解释在使用不同算法进行 ODE 求解时动作轨迹如何演化。

`示例：` 在下面的第一行中，VITA 仅经过一次 ODE 步骤就产生了结构化的动作轨迹，而传统的流匹配从高斯噪声开始并逐渐去噪。

<p align="center">
  <img src="https://raw.githubusercontent.com/ucd-dare/VITA/refs/heads/gh-pages/static/images/denoising.png" alt="VITA 去噪" />
</p>

---

### 🏋️ 训练

```bash
python flare/train.py policy=vita task=hook_package session=test
```

* 使用 `session` 命名检查点/日志（和 WandB 运行）。
* 默认配置：`./flare/configs/default_policy.yaml`
* 策略配置：`./flare/configs/policy/vita.yaml`
* 任务配置：`./flare/configs/task/hook_package.yaml`
* 当指定这些配置时，它们会覆盖默认值，例如 `policy=vita task=hook_package`。

根据需要覆盖标志：

```bash
# 示例 1：使用特定的 GPU
python flare/train.py policy=vita task=hook_package session=test device=cuda:2

# 示例 2：更改在线验证频率和回合数
python flare/train.py policy=vita task=hook_package session=test \
  val.val_online_freq=2000 val.eval_n_episodes=10

# 示例 3：运行消融实验
python flare/train.py policy=vita task=hook_package session=ablate \
  policy.vita.decode_flow_latents=False wandb.notes=ablation
```

#### 🎮 可用任务

可用的任务配置位于 `./flare/config/tasks`。要启动特定任务的训练，设置 `task` 标志（例如，`task=cube_transfer` 以加载 `cube_transfer.yaml`）。

```yaml
# AV-ALOHA 任务
cube_transfer
hook_package
pour_test_tube
slot_insertion
thread_needle
# Robomimic
robomimic_can
robomimic_square
# PushT
pusht
```

---

<p align="center">
  <img src=".assets/rollout.png" alt="VITA 运行" />
</p>

---

## 🌐 链接

* 🧪 [项目页面](https://ucd-dare.github.io/VITA/)
* 📄 [arXiv 论文](https://arxiv.org/abs/2507.13231)
* 📑 [PDF](https://arxiv.org/pdf/2507.13231)

我们衷心感谢启发 VITA 的开源代码库：
[AV-ALOHA](https://soltanilara.github.io/av-aloha/)、[Robomimic](https://robomimic.github.io/)、[LeRobot](https://github.com/huggingface/lerobot)、[CrossFlow](https://github.com/qihao067/CrossFlow)（[Qihao Liu](https://qihao067.github.io/)）！

---

## 📖 引用

```bibtex
@article{gao2025vita,
  title={VITA: Vision-to-Action Flow Matching Policy},
  author={Gao, Dechen and Zhao, Boqi and Lee, Andrew and Chuang, Ian and Zhou, Hanchu and Wang, Hang and Zhao, Zhe and Zhang, Junshan and Soltani, Iman},
  journal={arXiv preprint arXiv:2507.13231},
  year={2025}
}
```

---

