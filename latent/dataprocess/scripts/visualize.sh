#! /bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/init.sh"

usage() {
    echo "Usage: $0 <zarr_dataset_path> [output_dir]"
    echo "  zarr_dataset_path  Path to Zarr dataset (e.g. gym-av-aloha/outputs/dual_flip_box_video_50)"
    echo "  output_dir         Output directory (optional)"
    exit 1
}

[ "$1" = "-h" ] || [ "$1" = "--help" ] && usage

DATASET_PATH="$1"
OUTPUT_PATH="${2:-}"

if [ -z "$DATASET_PATH" ]; then
    echo "Error: Missing dataset path."
    usage
fi

get_abs_path() {
    local path=$1
    if [[ "$path" != /* ]]; then
        echo "$(pwd)/$path"
    else
        echo "$path"
    fi
}

DATASET_PATH=$(get_abs_path "$DATASET_PATH")
[ -n "$OUTPUT_PATH" ] && OUTPUT_PATH=$(get_abs_path "$OUTPUT_PATH")

if [ -n "$OUTPUT_PATH" ]; then
    python "$DATAPROCESS_DIR/visualize_zarr.py" "$DATASET_PATH" --output-dir "$OUTPUT_PATH"
else
    python "$DATAPROCESS_DIR/visualize_zarr.py" "$DATASET_PATH"
fi
