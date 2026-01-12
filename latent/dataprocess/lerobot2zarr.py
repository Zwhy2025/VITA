# Convert datasets from LeRobot to AV-ALOHA which is MUCH FASTER for training
#
# Usage:
# * Converting a dataset (local path or HuggingFace repo_id)
#       python dataprocess/lerobot2zarr.py datasets/r2v2/dual_flip_box_video_50/
#       python dataprocess/lerobot2zarr.py iantc104/av_aloha_sim_thread_needle
# * Display help message
#       python dataprocess/lerobot2zarr.py -h
#
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
from dataprocess.core.datasets import create_av_aloha_dataset_from_lerobot

# 确定输出根目录（使用环境变量或默认路径）
if 'FLARE_DATASETS_DIR' in os.environ:
    DEFAULT_OUTPUT_ROOT = Path(os.environ['FLARE_DATASETS_DIR'])
else:
    # 默认使用当前目录下的 outputs
    DEFAULT_OUTPUT_ROOT = Path(__file__).parent.parent / "gym-av-aloha" / "outputs"


def convert_dataset(repo_id: str):
    """
    转换数据集从 LeRobot 格式到 AV-ALOHA 格式。
    
    Args:
        repo_id: 数据集仓库 ID（HuggingFace）或本地路径（相对/绝对路径）
    """
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobotDataset is not available. Please install lerobot or provide dataset objects.")
        return
    
    # 尝试解析为本地路径
    path = Path(repo_id)
    if not path.is_absolute():
        # 如果是相对路径，基于当前工作目录解析
        path = Path.cwd() / path
    path = path.resolve()
    
    # 如果本地路径存在，使用本地路径；否则当作 HuggingFace repo_id
    if path.exists() and path.is_dir():
        repo_id = str(path)
        print(f"Using local path: {repo_id}")
    else:
        print(f"Using HuggingFace repo_id: {repo_id}")
    
    # 所有数据集使用默认值
    episodes = None  # 处理所有 episodes
    remove_keys = []  # 不移除任何键
    image_size = None  # 保持原始图像尺寸
    
    episodes_dict = {repo_id: episodes}
    
    # 确定输出路径：使用路径的最后一部分作为输出目录名
    # 从 repo_id 中提取最后一部分作为目录名
    if "/" in repo_id:
        safe_id = repo_id.split("/")[-1].strip("/")
    else:
        safe_id = repo_id.replace("\\", "/").split("/")[-1]
    
    output_root = DEFAULT_OUTPUT_ROOT / safe_id

    print(f"--- Converting Dataset: {repo_id} ---")
    print(f"Episodes to process: {'All' if episodes is None else len(episodes)}")
    print(f"Keys to remove: {remove_keys}")
    print(f"Target image size: {image_size}")
    print(f"Output directory: {output_root}")
    print("------------------------------------------")

    # 创建 LeRobotDataset 对象
    datasets = [LeRobotDataset(repo_id=r_id, episodes=ep) for r_id, ep in episodes_dict.items()]

    # 调用转换函数
    create_av_aloha_dataset_from_lerobot(
        datasets=datasets,
        root=output_root,
        remove_keys=remove_keys,
        image_size=image_size,
    )

    print(f"--- Successfully completed conversion for: {repo_id} ---")


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets from LeRobot format to AV-ALOHA format. "
                    "Supports both local paths and HuggingFace repo_ids.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "repo_id",
        type=str,
        metavar="REPO_ID_OR_PATH",
        help="Dataset repository ID (HuggingFace) or local path (relative/absolute).\n"
             "Examples:\n"
             "  - Local path: datasets/r2v2/dual_flip_box_video_50/\n"
             "  - HuggingFace: iantc104/av_aloha_sim_thread_needle",
    )

    args = parser.parse_args()
    convert_dataset(args.repo_id)


if __name__ == "__main__":
    main()
