#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 检查 conda 是否已安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 conda"
    exit 1
fi

# 初始化 conda（如果尚未初始化）
eval "$(conda shell.bash hook)"

# 创建 conda 环境（如果不存在）
if conda env list | grep -q "^vita "; then
    echo "警告: conda 环境 'vita' 已存在，跳过创建步骤"
else
    echo "创建 conda 环境 'vita'..."
    conda create --name vita python==3.10 -y
fi

# 激活 conda 环境
echo "激活 conda 环境 'vita'..."
conda activate vita

# 安装 cmake
echo "安装 cmake..."
conda install cmake -y

# 安装项目依赖
echo "安装项目依赖..."
pip install -e .
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "警告: requirements.txt 不存在，跳过"
fi

# 安装 LeRobot 依赖
echo "安装 LeRobot 依赖..."
cd lerobot
pip install -e .
cd "$SCRIPT_DIR"

# 安装 ffmpeg
echo "安装 ffmpeg..."
conda install -c conda-forge ffmpeg -y

# 修复 numpy/pandas 兼容性问题（如果存在）
echo "检查并修复 numpy/pandas 兼容性..."
pip install --upgrade --force-reinstall numpy pandas || true

# 设置数据集存储路径环境变量
FLARE_DATASETS_DIR="${SCRIPT_DIR}/gym-av-aloha/outputs"
ENV_EXPORT="export FLARE_DATASETS_DIR=${FLARE_DATASETS_DIR}"

# 检查是否已存在该环境变量设置，避免重复添加
if ! grep -q "FLARE_DATASETS_DIR.*gym-av-aloha/outputs" ~/.bashrc 2>/dev/null; then
    echo "设置环境变量 FLARE_DATASETS_DIR..."
    echo "$ENV_EXPORT" >> ~/.bashrc
    echo "已添加到 ~/.bashrc，请运行 'source ~/.bashrc' 或重新打开终端以生效"
else
    echo "环境变量 FLARE_DATASETS_DIR 已存在于 ~/.bashrc"
fi

# 在当前 shell 中设置环境变量（立即生效）
export FLARE_DATASETS_DIR="${FLARE_DATASETS_DIR}"

# 安装 AV-ALOHA 依赖
echo "安装 AV-ALOHA 依赖..."
cd gym-av-aloha
pip install -e .
cd "$SCRIPT_DIR"

echo ""
echo "✅ 安装完成！"
echo "环境变量 FLARE_DATASETS_DIR 已设置为: ${FLARE_DATASETS_DIR}"
echo "请确保 conda 环境 'vita' 已激活后再使用项目"