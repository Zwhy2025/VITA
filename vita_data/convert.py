# Convert datasets from LeRobot to AV-ALOHA which is MUCH FASTER for training
# 
# 独立实现版本 - 不调用项目内的任何库
# 原始代码备份（从dcc5e3d版本）
#
# Usage:
# * Listing all available datasets
#       python vita_data/convert.py -l
# * Converting a single task dataset
#       python vita_data/convert.py -r iantc104/av_aloha_sim_thread_needle
# * Display help message
#       python vita_data/convert.py -h
#
# 注意：此脚本需要从外部导入 LeRobotDataset 来创建数据集对象
# 实际使用时需要先创建 LeRobotDataset 对象，然后调用转换函数

import argparse
from pathlib import Path
import os

# 导入标准库中的 LeRobotDataset（如果可用）
# 如果不可用，需要从外部传入
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    LeRobotDataset = None

# 导入本地的转换函数
from vita_data.datasets import create_av_aloha_dataset_from_lerobot

# 确定输出根目录（使用环境变量或默认路径）
if 'FLARE_DATASETS_DIR' in os.environ:
    DEFAULT_OUTPUT_ROOT = Path(os.environ['FLARE_DATASETS_DIR'])
else:
    # 默认使用当前目录下的 outputs
    DEFAULT_OUTPUT_ROOT = Path(__file__).parent.parent / "gym-av-aloha" / "outputs"


DATASET_CONFIGS = {
    # gym-av-aloha tasks
    "iantc104/av_aloha_sim_cube_transfer": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_thread_needle": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_pour_test_tube": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_slot_insertion": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    "iantc104/av_aloha_sim_hook_package": {
        "episodes": list(range(0, 100)),
        "remove_keys": [
            "observation.images.wrist_cam_left",
            "observation.images.wrist_cam_right",
            "observation.images.worms_eye_cam",
            "observation.images.overhead_cam",
        ],
        "image_size": (240, 320),
    },
    # robomimic tasks
    "iantc104/robomimic_sim_square": {
        "episodes": list(range(0, 174)),
        "remove_keys": ["observation.images.robot0_eye_in_hand"],
        "image_size": (256, 256),
    },
    "iantc104/robomimic_sim_can": {
        "episodes": list(range(0, 191)),
        "remove_keys": ["observation.images.robot0_eye_in_hand"],
        "image_size": (256, 256),
    },
    # pusht
    "lerobot/pusht": {
        "episodes": list(range(0, 206)),
        "remove_keys": [],
        "image_size": (96, 96),
    },
}


def list_datasets():
    print("--- Available Dataset Repository IDs (Repo IDs) ---")
    for repo_id in DATASET_CONFIGS:
        print(f"  - {repo_id}")
    print("--------------------------------------------------")


def convert_dataset(repo_id: str, output_root: Path | None = None):
    """
    原始版本的convert_dataset函数（从dcc5e3d版本备份）
    
    此函数完全独立，使用本地的实现，不调用项目内的库。
    """
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobotDataset is not available. Please install lerobot or provide dataset objects.")
        return
    
    if repo_id not in DATASET_CONFIGS:
        print(f"Error: Repository ID '{repo_id}' not found in configurations.")
        list_datasets()
        return

    config = DATASET_CONFIGS[repo_id]
    episodes_dict = {repo_id: config["episodes"]}
    
    # 确定输出路径
    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT / repo_id
    else:
        output_root = Path(output_root) / repo_id

    print(f"--- Converting Dataset: {repo_id} ---")
    print(f"Episodes to process: {len(config['episodes'])}")
    print(f"Keys to remove: {config['remove_keys']}")
    print(f"Target image size: {config['image_size']}")
    print(f"Output directory: {output_root}")
    print("------------------------------------------")

    # 创建 LeRobotDataset 对象
    datasets = [LeRobotDataset(repo_id=r_id, episodes=episodes) for r_id, episodes in episodes_dict.items()]

    # 调用转换函数
    create_av_aloha_dataset_from_lerobot(
        datasets=datasets,
        root=output_root,
        remove_keys=config["remove_keys"],
        image_size=config["image_size"],
    )

    print(f"--- Successfully completed conversion for: {repo_id} ---")


def main():
    parser = argparse.ArgumentParser(
        description="A script to convert AV-ALOHA and Robomimic datasets from Hugging Face.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-l", "--ls",
        action="store_true",
        help="List all available dataset repository IDs (repo IDs).",
    )
    group.add_argument(
        "-r", "--repo",
        type=str,
        metavar="REPO_ID",
        help="Specify the single dataset REPO_ID to convert (e.g., iantc104/robomimic_sim_transport).",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        metavar="OUTPUT_DIR",
        help="Output directory for converted dataset (default: FLARE_DATASETS_DIR or ./gym-av-aloha/outputs).",
    )

    args = parser.parse_args()

    if args.ls:
        list_datasets()
    elif args.repo:
        output_root = Path(args.output) if args.output else None
        convert_dataset(args.repo, output_root=output_root)


if __name__ == "__main__":
    main()
