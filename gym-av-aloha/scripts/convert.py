# Convert datasets from LeRobot to AV-ALOHA which is MUCH FASTER for training
# Usage:
# * Listing all available datasets
#       python convert.py -l
# * Converting a single task dataset
#       python convert.py -r iantc104/av_aloha_sim_thread_needle
# * Converting a local dataset
#       python convert.py -r /path/to/dataset
# * Converting with parallel processing (8 workers, optimized with direct Zarr writes)
#       python convert.py -r /path/to/dataset -w 8
# * Converting with GPU acceleration for image resizing
#       python convert.py -r /path/to/dataset -w 8 --gpu
# * Display help message
#       python convert.py -h

import argparse
import os
from pathlib import Path
from gym_av_aloha.datasets.av_aloha_dataset import (
    create_av_aloha_dataset_from_lerobot,
    create_av_aloha_dataset_from_lerobot_parallel,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


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


def convert_dataset(repo_id_or_path: str, num_workers: int = 1, use_gpu: bool = False):
    """
    Convert a dataset from LeRobot format to AV-ALOHA format.
    
    Args:
        repo_id_or_path: Either a repository ID (e.g., 'iantc104/av_aloha_sim_cube_transfer')
                         or a local path to a dataset directory
        num_workers: Number of parallel workers for conversion (default: 1 for serial)
        use_gpu: Whether to use GPU for image resizing (default: False)
    """
    # Check if it's a local path
    dataset_path = Path(repo_id_or_path)
    is_local_path = dataset_path.exists() and dataset_path.is_dir()
    
    # Choose conversion function based on num_workers
    convert_func = (
        create_av_aloha_dataset_from_lerobot_parallel 
        if num_workers > 1 
        else create_av_aloha_dataset_from_lerobot
    )
    
    if is_local_path:
        # It's a local path, load metadata to get dataset info
        try:
            # Try to load metadata from the local path
            # Use a dummy repo_id and force_cache_sync to avoid downloading from hub
            dummy_repo_id = "local_dataset"
            meta = LeRobotDatasetMetadata(dummy_repo_id, root=dataset_path, force_cache_sync=False)
            total_episodes = meta.total_episodes
            
            # Generate a repo_id from the path (use the last two parts)
            path_parts = dataset_path.parts
            if len(path_parts) >= 2:
                repo_id = f"{path_parts[-2]}/{path_parts[-1]}"
            else:
                repo_id = dataset_path.name
            
            # Default configuration for local datasets
            config = {
                "episodes": list(range(0, total_episodes)),
                "remove_keys": [],  # Don't remove any keys by default
                "image_size": None,  # Keep original size by default
            }
            
            print(f"--- Converting Local Dataset: {dataset_path} ---")
            print(f"Repository ID (generated): {repo_id}")
            print(f"Total episodes: {total_episodes}")
            print(f"Episodes to process: {len(config['episodes'])}")
            print(f"Keys to remove: {config['remove_keys']}")
            print(f"Target image size: {config['image_size'] or 'Original size'}")
            print(f"Parallel workers: {num_workers}")
            print(f"GPU acceleration: {use_gpu}")
            print("------------------------------------------")
            
            episodes_dict = {repo_id: config["episodes"]}
            
            # For local datasets, we need to pass the dataset root to LeRobotDataset
            # LeRobotDataset expects root to point directly to the dataset directory
            # So we pass the dataset_path itself as the root, and use a simple repo_id
            # Actually, LeRobotDataset with root expects root to be the parent directory
            # and repo_id to be the dataset name relative to root
            # So if dataset is at /path/to/datasets/ur12e/real_libero_spatial/
            # root should be /path/to/datasets/ and repo_id should be ur12e/real_libero_spatial
            dataset_root = dataset_path.parent.parent  # Go up two levels to get the datasets root
            if len(path_parts) >= 2:
                # repo_id should be relative to dataset_root
                repo_id_for_lerobot = f"{path_parts[-2]}/{path_parts[-1]}"
            else:
                repo_id_for_lerobot = dataset_path.name
            
            if num_workers > 1:
                convert_func(
                    episodes=episodes_dict,
                    repo_id=repo_id_for_lerobot,
                    dataset_root=dataset_root,
                    remove_keys=config["remove_keys"],
                    image_size=config["image_size"],
                    num_workers=num_workers,
                    use_gpu=use_gpu,
                )
            else:
                convert_func(
                    episodes=episodes_dict,
                    repo_id=repo_id_for_lerobot,
                    dataset_root=dataset_root,
                    remove_keys=config["remove_keys"],
                    image_size=config["image_size"],
                )
            
            print(f"--- Successfully completed conversion for: {dataset_path} ---")
            return
            
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            print("Make sure the path contains a valid LeRobot dataset with meta/ directory.")
            raise
    
    # It's a repo_id, check if it's in configurations
    if repo_id_or_path not in DATASET_CONFIGS:
        print(f"Error: Repository ID '{repo_id_or_path}' not found in configurations.")
        print("If you're trying to use a local path, make sure it exists and is a valid directory.")
        list_datasets()
        return

    config = DATASET_CONFIGS[repo_id_or_path]
    repo_id = repo_id_or_path

    episodes_dict = {repo_id: config["episodes"]}

    print(f"--- Converting Dataset: {repo_id} ---")
    print(f"Episodes to process: {len(config['episodes'])}")
    print(f"Keys to remove: {config['remove_keys']}")
    print(f"Target image size: {config['image_size']}")
    print(f"Parallel workers: {num_workers}")
    print(f"GPU acceleration: {use_gpu}")
    print("------------------------------------------")

    if num_workers > 1:
        convert_func(
            episodes=episodes_dict,
            repo_id=repo_id,
            remove_keys=config["remove_keys"],
            image_size=config["image_size"],
            num_workers=num_workers,
            use_gpu=use_gpu,
        )
    else:
        convert_func(
            episodes=episodes_dict,
            repo_id=repo_id,
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
        "-w", "--workers",
        type=int,
        default=8,
        metavar="NUM",
        help="Number of parallel workers for conversion (default: 8).\n"
             "Uses optimized direct Zarr writes, eliminating merge overhead.",
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration for image resizing.\n"
             "Workers are automatically distributed across available GPUs.",
    )

    args = parser.parse_args()

    if args.ls:
        list_datasets()
    elif args.repo:
        convert_dataset(args.repo, num_workers=args.workers, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
