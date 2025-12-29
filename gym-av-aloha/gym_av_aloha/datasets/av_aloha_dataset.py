import torch
from pathlib import Path
import os
import numpy as np
from typing import Callable
import gym_av_aloha
from gym_av_aloha.common.replay_buffer import ReplayBuffer
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_episode_data_index,
    check_timestamps_sync,
)
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from tqdm import tqdm
from lerobot.common.datasets.compute_stats import aggregate_stats
import shutil
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

ROOT = Path(os.path.dirname(os.path.dirname(gym_av_aloha.__file__))) / "outputs"

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
    episodes: dict[str, list[int]] | None = None,
    repo_id: str | None = None,
    root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
):
    output_root = Path(root) if root else ROOT / repo_id
    # create lerobot datasets
    # If dataset_root is provided, use it for loading datasets; otherwise use default location
    # Force using pyav backend to avoid torchcodec loading issues
    if dataset_root:
        datasets = [LeRobotDataset(repo_id=r_id, root=Path(dataset_root) / r_id, episodes=episodes, video_backend="pyav") for r_id, episodes in episodes.items()]
    else:
        datasets = [LeRobotDataset(repo_id=r_id, episodes=episodes, video_backend="pyav") for r_id, episodes in episodes.items()]
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
    if output_root.exists():
        print(f"Removing existing directory {output_root}...")
        shutil.rmtree(output_root)

    # create new replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=output_root, mode="a")
    # metadata
    config = {
        "repo_id": dataset.repo_id,
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
    config_path = output_root / "config.json"
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
    print(f"Converted dataset saved to {output_root}.")


def _process_episode_batch_worker(args):
    """
    Worker function to process a batch of episodes in a separate process.
    
    Args:
        args: Tuple containing all necessary parameters for processing
    
    Returns:
        Tuple of (worker_id, temp_dir, num_episodes_processed, total_frames)
    """
    (
        worker_id,
        episode_indices,
        repo_id,
        dataset_root,
        temp_dir,
        features,
        tasks_reversed,
        image_size,
        remove_keys,
    ) = args
    
    # Create dataset in this process
    episodes_dict = {repo_id: episode_indices}
    if dataset_root:
        dataset = LeRobotDataset(
            repo_id=repo_id, 
            root=Path(dataset_root) / repo_id, 
            episodes=episode_indices, 
            video_backend="pyav"
        )
    else:
        dataset = LeRobotDataset(
            repo_id=repo_id, 
            episodes=episode_indices, 
            video_backend="pyav"
        )
    
    # Create temporary replay buffer
    temp_path = Path(temp_dir)
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=temp_path, mode="a")
    
    def convert(k, v: torch.Tensor):
        dtype = features[k]['dtype']
        if dtype in ['image', 'video']:
            if image_size is not None:
                v = Resize(image_size)(v)
            v = v.permute(0, 2, 3, 1)
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v
    
    total_frames = 0
    local_episode_idx = 0
    
    for i in range(dataset.num_episodes):
        from_idx = dataset.episode_data_index['from'][i]
        to_idx = dataset.episode_data_index['to'][i]
        subset = Subset(dataset, range(from_idx, to_idx))
        # Use fewer workers in subprocess to avoid resource contention
        dataloader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=2)
        
        data = []
        for batch in dataloader:
            if 'task_index' in batch:
                batch['task_index'] = torch.tensor(
                    [tasks_reversed[k] for k in batch['task']], dtype=int
                )
                del batch["task"]
            batch['episode_index'] = torch.full_like(batch['episode_index'], local_episode_idx)
            data.append(batch)
        
        batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
        batch = {k: convert(k, v) for k, v in batch.items() if k in features}
        replay_buffer.add_episode(batch, compressors='disk')
        
        total_frames += to_idx - from_idx
        local_episode_idx += 1
    
    return (worker_id, str(temp_path), local_episode_idx, total_frames)


def _merge_zarr_buffers(temp_dirs: list[str], output_root: Path, config: dict):
    """
    Merge multiple temporary zarr buffers into the final output.
    
    Args:
        temp_dirs: List of paths to temporary zarr directories (in order)
        output_root: Path to the final output directory
        config: Configuration dict to save
    """
    # Create final replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=output_root, mode="a")
    
    episode_idx = 0
    for temp_dir in temp_dirs:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            continue
            
        # Load temporary buffer
        temp_buffer = ReplayBuffer.copy_from_path(temp_path)
        
        # Copy episodes from temp buffer to final buffer
        for i in range(temp_buffer.n_episodes):
            episode_data = temp_buffer.get_episode(i, copy=True)
            # Update episode_index to be sequential
            episode_data['episode_index'] = np.full_like(
                episode_data['episode_index'], episode_idx
            )
            replay_buffer.add_episode(episode_data, compressors='disk')
            episode_idx += 1
    
    # Save config
    config_path = output_root / "config.json"
    with open(config_path, "w") as f:
        json.dump(make_json_serializable(config), f, indent=4)
    
    print(f"Merged {episode_idx} episodes into {output_root}")


def create_av_aloha_dataset_from_lerobot_parallel(
    episodes: dict[str, list[int]] | None = None,
    repo_id: str | None = None,
    root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
    num_workers: int = 4,
):
    """
    Parallel version of create_av_aloha_dataset_from_lerobot.
    
    Processes episodes in parallel using multiple worker processes,
    then merges the results into a single output.
    
    Args:
        episodes: Dict mapping repo_id to list of episode indices
        repo_id: Repository ID for the dataset
        root: Output root directory
        dataset_root: Root directory for input datasets
        image_size: Target image size (H, W) or None to keep original
        remove_keys: List of keys to remove from the dataset
        num_workers: Number of parallel workers (default: 4)
    """
    output_root = Path(root) if root else ROOT / repo_id
    
    # If only 1 worker, use the original serial function
    if num_workers <= 1:
        return create_av_aloha_dataset_from_lerobot(
            episodes=episodes,
            repo_id=repo_id,
            root=root,
            dataset_root=dataset_root,
            image_size=image_size,
            remove_keys=remove_keys,
        )
    
    print(f"Starting parallel conversion with {num_workers} workers...")
    
    # Load metadata from the first dataset to get configuration
    if dataset_root:
        datasets = [
            LeRobotDataset(
                repo_id=r_id, 
                root=Path(dataset_root) / r_id, 
                episodes=eps, 
                video_backend="pyav"
            ) 
            for r_id, eps in episodes.items()
        ]
    else:
        datasets = [
            LeRobotDataset(repo_id=r_id, episodes=eps, video_backend="pyav") 
            for r_id, eps in episodes.items()
        ]
    
    # Collect metadata (same as serial version)
    disabled_features = set()
    intersection_features = set(datasets[0].features)
    for ds in datasets:
        intersection_features.intersection_update(ds.features)
    if len(intersection_features) == 0:
        raise RuntimeError(
            "Multiple datasets were provided but they had no keys common to all of them."
        )
    for ds in datasets:
        extra_keys = set(ds.features).difference(intersection_features)
        if len(extra_keys) > 0:
            print(f"keys {extra_keys} of {ds.repo_id} were disabled")
        disabled_features.update(extra_keys)
    
    print(f"Disabled features: {disabled_features}.\n")
    
    fps = datasets[0].meta.fps
    assert all(dataset.meta.fps == fps for dataset in datasets)
    
    num_frames = sum(d.num_frames for d in datasets)
    num_episodes = sum(d.num_episodes for d in datasets)
    
    features = {}
    for dataset in datasets:
        features.update({k: v for k, v in dataset.features.items()})
    features = {k: v for k, v in features.items() if k not in disabled_features}
    features = {k: v for k, v in features.items() if k not in remove_keys}
    
    camera_keys = set()
    video_keys = set()
    image_keys = set()
    for dataset in datasets:
        camera_keys.update(dataset.meta.camera_keys)
        video_keys.update(dataset.meta.video_keys)
        image_keys.update(dataset.meta.image_keys)
    camera_keys = [k for k in camera_keys if k in features]
    video_keys = [k for k in video_keys if k in features]
    image_keys = [k for k in image_keys if k in features]
    
    episodes_stats = []
    for dataset in datasets:
        ep = dataset.episodes if dataset.episodes else range(dataset.num_episodes)
        for ep_idx in ep:
            episodes_stats.append({
                k: v for k, v in dataset.meta.episodes_stats[ep_idx].items() 
                if k in features
            })
    stats = aggregate_stats(episodes_stats)
    
    tasks = []
    for ds in datasets:
        tasks.extend(ds.meta.tasks.values())
    tasks = {i: task for i, task in enumerate(tasks)}
    tasks_reversed = {v: k for k, v in tasks.items()}
    
    # Prepare config
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
    
    # Clear datasets to free memory before spawning workers
    del datasets
    
    # Remove old output if exists
    if output_root.exists():
        print(f"Removing existing directory {output_root}...")
        shutil.rmtree(output_root)
    
    # Create temporary directory for worker outputs
    temp_base = tempfile.mkdtemp(prefix="av_aloha_convert_")
    print(f"Using temporary directory: {temp_base}")
    
    try:
        # Distribute episodes across workers
        # For simplicity, we handle single repo_id case (most common)
        main_repo_id = list(episodes.keys())[0]
        all_episodes = episodes[main_repo_id]
        
        # Split episodes into chunks for each worker
        chunk_size = max(1, len(all_episodes) // num_workers)
        episode_chunks = []
        for i in range(0, len(all_episodes), chunk_size):
            chunk = all_episodes[i:i + chunk_size]
            if chunk:
                episode_chunks.append(chunk)
        
        # Adjust num_workers if we have fewer chunks
        actual_workers = min(num_workers, len(episode_chunks))
        print(f"Distributing {len(all_episodes)} episodes across {actual_workers} workers")
        
        # Prepare worker arguments
        worker_args = []
        for worker_id, chunk in enumerate(episode_chunks):
            temp_dir = os.path.join(temp_base, f"worker_{worker_id}")
            worker_args.append((
                worker_id,
                chunk,
                main_repo_id,
                str(dataset_root) if dataset_root else None,
                temp_dir,
                features,
                tasks_reversed,
                image_size,
                remove_keys,
            ))
        
        # Process in parallel
        temp_dirs = [None] * len(worker_args)
        
        # Use fork on Linux (faster, avoids re-importing modules)
        # Use spawn on other platforms for safety
        import platform
        if platform.system() == 'Linux':
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_process_episode_batch_worker, args): args[0] 
                for args in worker_args
            }
            
            with tqdm(total=len(futures), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        result = future.result()
                        w_id, temp_path, n_eps, n_frames = result
                        temp_dirs[w_id] = temp_path
                        pbar.set_postfix({
                            f"Worker {w_id}": f"{n_eps} eps, {n_frames} frames"
                        })
                        pbar.update(1)
                    except Exception as e:
                        print(f"Worker {worker_id} failed with error: {e}")
                        raise
        
        # Merge results
        print("Merging temporary buffers...")
        # Filter out None values and sort by worker_id
        valid_temp_dirs = [d for d in temp_dirs if d is not None]
        _merge_zarr_buffers(valid_temp_dirs, output_root, config)
        
    finally:
        # Cleanup temporary directory
        print(f"Cleaning up temporary directory: {temp_base}")
        shutil.rmtree(temp_base, ignore_errors=True)
    
    print(f"Parallel conversion complete. Dataset saved to {output_root}")


def get_dataset_config(
    repo_id: str | None = None,
    root: str | Path | None = None,
) -> LeRobotDatasetMetadata:
    root = Path(root) if root else ROOT / repo_id
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Please create the dataset first.")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

class AVAlohaDatasetMeta():
    def __init__(self, repo_id: str | None = None, root: str | Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root) if root else ROOT / repo_id
        self.config = get_dataset_config(repo_id=self.repo_id, root=self.root)

        # convert config['tasks'] keys to int, not string
        if 'tasks' in self.config:
            self.config['tasks'] = {int(k): v for k, v in self.config['tasks'].items()}

    @property
    def stats(self) -> dict:
        return self.config['stats']
    
    @property
    def num_frames(self) -> int:
        return self.config['num_frames']
    
    @property
    def num_episodes(self) -> int:
        return self.config['num_episodes']
    
    @property
    def features(self):
        return self.config['features']
    
    @property
    def camera_keys(self):
        return self.config['camera_keys']
    
    @property
    def video_keys(self):
        return self.config['video_keys']
    
    @property
    def image_keys(self):
        return self.config['image_keys']
    
    @property
    def fps(self) -> float:
        return self.config['fps']
    
    @property
    def tasks(self):
        return self.config['tasks']

class AVAlohaDataset(torch.utils.data.Dataset):
    def __init__(self,
                 repo_id: str | None = None,
                 root: str | Path | None = None,
                 episodes: list[int] | None = None,
                 image_transforms: Callable | None = None,
                 delta_timestamps: dict[list[float]] | None = None,
                 tolerance_s: float = 1e-4,
                 ):
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root) if root else ROOT / repo_id
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.episodes = episodes

        # create zarr dataset + lerobot metadata
        self.replay_buffer = ReplayBuffer.copy_from_path(self.root)
        # self.replay_buffer = ReplayBuffer.create_from_path(self.root, mode='r')
        self.meta = AVAlohaDatasetMeta(repo_id=self.repo_id, root=self.root)

        # if no episodes are specified, use all episodes in the replay buffer
        if not self.episodes: 
            self.episodes = list(range(self.meta.num_episodes))

        # calculate length of the dataset
        self.length = sum([self.replay_buffer.episode_lengths[i] for i in self.episodes])

        # add task index to delta timestamps
        if 'task_index' in self.features:
            self.delta_timestamps['task_index'] = [0]  

        # from and to indices for episodes
        self.episode_data_index = get_episode_data_index({
            i: {'length': length}
            for i, length in enumerate(self.replay_buffer.episode_lengths)
        }, self.episodes)

        # Check timestamps
        timestamps = np.array(self.replay_buffer['timestamp'])
        episode_indices = np.array(self.replay_buffer['episode_index'])
        # keep only timestamps and episode_indices for the selected episodes
        if self.episodes is not None:
            mask = np.isin(episode_indices, self.episodes)
            timestamps = timestamps[mask]
            episode_indices = episode_indices[mask]
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_replay_buffer(self, query_indices: list[str, list[int]]) -> dict:
        return {
            key: self.replay_buffer[key][q_idx]
            for key, q_idx in query_indices.items()
        }

    @property
    def stats(self):
        return self.meta.stats

    @property
    def features(self):
        return self.meta.features

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return self.meta.num_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.num_episodes

    @property
    def video_keys(self):
        return self.meta.video_keys

    @property
    def image_keys(self):
        return self.meta.image_keys

    @property
    def camera_keys(self):
        return self.meta.camera_keys
    
    @property
    def tasks(self):
        return self.meta.tasks

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx = self.replay_buffer["episode_index"][idx]
        item = {"episode_index": torch.tensor(ep_idx)}

        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_replay_buffer(query_indices)
        item = {**item, **padding}
        for key, val in query_result.items():
            if key in self.image_keys or key in self.video_keys:
                item[key] = torch.from_numpy(val).type(torch.float32).permute(0, 3, 1, 2) / 255.0
            else:
                item[key] = torch.from_numpy(val)

        if self.image_transforms is not None:
            image_keys = self.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        if "task_index" in item:
            task_idx = item["task_index"].item()
            item["task"] = self.tasks[task_idx]

        return item
