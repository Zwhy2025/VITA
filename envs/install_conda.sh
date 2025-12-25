#!/bin/bash
set -e  # 遇到错误立即退出
export DEBIAN_FRONTEND=noninteractive

# 配置参数
# 根据架构选择Miniconda版本
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        MINICONDA_VERSION="Miniconda3-latest-Linux-x86_64.sh"
        ;;
    aarch64|arm64)
        MINICONDA_VERSION="Miniconda3-latest-Linux-aarch64.sh"
        ;;
    *)
        echo "错误：不支持的架构: $ARCH"
        exit 1
        ;;
esac

MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/${MINICONDA_VERSION}"
INSTALL_PATH="${HOME}/miniconda3"
ENV_NAME="py3.10"
PYTHON_VERSION="3.10"

# 检查系统是否已安装conda
if [ -d "${INSTALL_PATH}" ]; then
    echo "检测到已安装Miniconda，将直接配置环境..."
else
    # 下载Miniconda安装包
    echo "正在从清华镜像下载Miniconda..."
    if command -v wget &> /dev/null; then
        wget -q "${MINICONDA_URL}" -O "${MINICONDA_VERSION}"
    elif command -v curl &> /dev/null; then
        curl -sSL "${MINICONDA_URL}" -o "${MINICONDA_VERSION}"
    else
        echo "错误：未找到wget或curl，请先安装其中一个工具"
        exit 1
    fi

    # 执行安装
    echo "开始安装Miniconda..."
    bash "${MINICONDA_VERSION}" -b -p "${INSTALL_PATH}"
    
    # 清理安装包
    rm -f "${MINICONDA_VERSION}"
fi

# 初始化conda
echo "初始化conda环境..."
"${INSTALL_PATH}/bin/conda" init bash

# 配置conda镜像源（清华）
echo "配置conda国内镜像源..."
cat << EOF > "${HOME}/.condarc"
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# 配置pip镜像源
echo "配置pip国内镜像源..."
mkdir -p "${HOME}/.pip"
cat << EOF > "${HOME}/.pip/pip.conf"
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 创建Python 3.10环境
echo "创建${ENV_NAME}环境（Python ${PYTHON_VERSION}）..."
"${INSTALL_PATH}/bin/conda" create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# 完成提示
echo "=============================================="
echo "安装配置完成！"
echo "请执行以下命令使配置生效："
echo "  source ~/.bashrc"
echo "激活环境："
echo "  conda activate ${ENV_NAME}"
echo "=============================================="