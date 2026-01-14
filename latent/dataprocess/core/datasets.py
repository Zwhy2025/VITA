"""
数据转换模块
"""
import torch
from pathlib import Path
import os
import numpy as np
from typing import Callable
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from tqdm import tqdm
import shutil
import json

# 使用本地的 ReplayBuffer 和 compute_stats
from .replay_buffer import ReplayBuffer
from .compute_stats import aggregate_stats


def make_json_serializable(obj):
    """Convert an object to a JSON-serializable format."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_av_aloha_dataset_from_lerobot(
    datasets,  # LeRobotDataset 对象列表，从外部传入
    root: str | Path,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
):
    """
    将 LeRobot 格式的数据集转换为 AV-ALOHA 格式。
    
    Args:
        datasets: LeRobotDataset 对象列表，需要从外部创建并传入
        root: 输出根目录
        image_size: 目标图像大小
        remove_keys: 要移除的键列表
    """
    root = Path(root)
    
    # Disable any data keys that are not common across all of the datasets.
    disabled_features = set()
    intersection_features = set(datasets[0].features)
    for ds in datasets:
        intersection_features.intersection_update(ds.features)
    if len(intersection_features) == 0:
        raise RuntimeError(
            "Multiple datasets were provided but they had no keys common to all of them. "
            "The multi-dataset functionality currently only keeps common keys."
        )
    for ds in datasets:
        extra_keys = set(ds.features).difference(intersection_features)
        if len(extra_keys) > 0:
            print(
                f"keys {extra_keys} of {ds.repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
        disabled_features.update(extra_keys)
    print(
        f"Disabled features: {disabled_features}.\n"
    )
    # fps
    fps = datasets[0].meta.fps
    assert all(dataset.meta.fps == fps for dataset in datasets), "Datasets have different fps values."
    # num frames
    num_frames = sum(d.num_frames for d in datasets)
    # num episodes
    num_episodes = sum(d.num_episodes for d in datasets)
    # features
    features = {}
    for dataset in datasets:
        features.update({k: v for k, v in dataset.features.items()})
    features = {k: v for k, v in features.items() if k not in disabled_features}
    features = {k: v for k, v in features.items() if k not in remove_keys}
    # camera keys
    camera_keys = set([])
    for dataset in datasets:
        camera_keys.update(dataset.meta.camera_keys)
    camera_keys = [k for k in camera_keys if k in features]
    # video keys
    video_keys = set([])
    for dataset in datasets:
        video_keys.update(dataset.meta.video_keys)
    video_keys = [k for k in video_keys if k in features]
    # image keys
    image_keys = set([])
    for dataset in datasets:
        image_keys.update(dataset.meta.image_keys)
    image_keys = [k for k in image_keys if k in features]
    # stats
    episodes_stats = []
    for dataset in datasets:
        ep = dataset.episodes if dataset.episodes else range(dataset.num_episodes)
        for ep_idx in ep:
            episodes_stats.append({k: v for k, v in dataset.meta.episodes_stats[ep_idx].items() if k in features})
    stats = aggregate_stats(episodes_stats)
    # tasks
    tasks = []
    for ds in datasets:
        tasks.extend(ds.meta.tasks.values())
    tasks = {i: task for i, task in enumerate(tasks)}
    tasks_reversed = {v: k for k, v in tasks.items()}

    # remove old replay buffer if it exists
    if root.exists():
        print(f"Removing existing directory {root}...")
        shutil.rmtree(root)

    # create new replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=root, mode="a")
    # metadata
    config = {
        "repo_id": datasets[0].repo_id,
        "stats": stats,
        "num_frames": num_frames,
        "num_episodes": num_episodes,
        "features": features,
        "camera_keys": camera_keys,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "fps": fps,
        "tasks": tasks,
    }
    config_path = root / "config.json"
    with open(config_path, "w") as f: 
        json.dump(make_json_serializable(config), f, indent=4)
        
    def convert(k, v: torch.Tensor):
        dtype = features[k]['dtype']
        if dtype in ['image', 'video']:
            if image_size is not None:
                v = Resize(image_size)(v)
            # (B, C, H, W) to (B, H, W, C)
            v = v.permute(0, 2, 3, 1)
            # convert from torch float32 to numpy uint8
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v
        
    # iterate through dataset
    episode_idx = 0
    for dataset in datasets:
        for i in range(dataset.num_episodes):
            print(f"Converting episode {episode_idx}...")
            from_idx = dataset.episode_data_index['from'][i]
            to_idx = dataset.episode_data_index['to'][i]
            subset = Subset(dataset, range(from_idx, to_idx))
            dataloader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=8)
            data = []
            for batch in tqdm(dataloader):
                if 'task_index' in batch:
                    batch['task_index'] = torch.tensor([tasks_reversed[k] for k in batch['task']], dtype=int)
                    del batch["task"]
                batch['episode_index'] = torch.full_like(batch['episode_index'], episode_idx)
                data.append(batch)
            # since batch is a dict go through keys and cat them into a batch
            batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
            assert batch['action'].shape[0] == to_idx - from_idx, f"Batch size does not match episode length. Expected {to_idx - from_idx}, got {batch['action'].shape[0]}."
            batch = {k:convert(k,v) for k,v in batch.items() if k in features}
            replay_buffer.add_episode(batch, compressors='disk')
            print(f"Episode {episode_idx} converted and added to replay buffer.")
            episode_idx += 1
    print(f"Converted dataset saved to {root}.")
