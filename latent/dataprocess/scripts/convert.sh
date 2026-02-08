#! /bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/init.sh"

usage() {
    echo "Usage: $0 <type> <input_path> [output_path]"
    echo "  type         l2z (LeRobot to Zarr) or m2z (MCAP to Zarr)"
    echo "  input_path   Dataset path (relative or absolute)"
    echo "  output_path  Output directory (optional)"
    exit 1
}

[ "$1" = "-h" ] || [ "$1" = "--help" ] && usage

ORIGINAL_TYPE="$1"
DATASET_PATH="$2"
OUTPUT_PATH="${3:-}"

if [ -z "$ORIGINAL_TYPE" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# 绝对路径处理
get_abs_path() {
    local path=$1
    if [[ "$path" != /* ]]; then
        echo "$(pwd)/$path"
    else
        echo "$path"
    fi
}

DATASET_PATH=$(get_abs_path "$DATASET_PATH")

if [ -n "$OUTPUT_PATH" ]; then
    OUTPUT_PATH=$(get_abs_path "$OUTPUT_PATH")
fi

if [ "$ORIGINAL_TYPE" == "l2z" ]; then
    if [ -n "$OUTPUT_PATH" ]; then
        python "$DATAPROCESS_DIR/lerobot2zarr.py" "$DATASET_PATH" --output "$OUTPUT_PATH"
    else
        python "$DATAPROCESS_DIR/lerobot2zarr.py" "$DATASET_PATH"
    fi
elif [ "$ORIGINAL_TYPE" == "m2z" ]; then
    python "$DATAPROCESS_DIR/mcap2zarr.py" "$DATASET_PATH"
else
    echo "Invalid original type: $ORIGINAL_TYPE"
    exit 1
fi



