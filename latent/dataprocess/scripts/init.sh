#! /bin/bash
# 环境与依赖初始化，供 convert.sh / visualize.sh 等 source 使用

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATAPROCESS_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$DATAPROCESS_DIR/../.." && pwd )"
export SCRIPT_DIR DATAPROCESS_DIR PROJECT_ROOT

eval "$(conda shell.bash hook)"
ENV_NAME="convert"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    conda activate "$ENV_NAME"
fi

if [ -f "$DATAPROCESS_DIR/requirements.txt" ]; then
    pip install -q -r "$DATAPROCESS_DIR/requirements.txt" > /dev/null 2>&1
fi

cd "$PROJECT_ROOT"
