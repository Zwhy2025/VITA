from __future__ import annotations

from pathlib import Path
import io
import json
import shutil
from typing import Iterable
import logging

import numpy as np
import pyarrow.parquet as pq
import av
from PIL import Image
from tqdm import tqdm

from dataprocess.core.replay_buffer import ReplayBuffer
from dataprocess.core.compute_stats import aggregate_stats


logger = logging.getLogger(__name__)
INFO_PATH = "meta/info.json"

EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

DEFAULT_CHUNK_SIZE = 1000

DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int64": np.int64,
    "int32": np.int32,
    "bool": np.bool_,
}


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonlines(path: Path) -> list[dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        cursor = outdict
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return outdict


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    flat = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(flat)

def load_info(root: Path) -> dict:
    info = load_json(root / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def load_tasks(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    path = root / TASKS_PATH
    if not path.exists():
        return {}, {}
    tasks_raw = load_jsonlines(path)
    tasks_raw = sorted(tasks_raw, key=lambda x: x["task_index"])
    tasks = {int(item["task_index"]): item["task"] for item in tasks_raw}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def load_episodes(root: Path) -> dict[int, dict]:
    episodes = load_jsonlines(root / EPISODES_PATH)
    episodes = sorted(episodes, key=lambda x: x["episode_index"])
    return {int(item["episode_index"]): item for item in episodes}


def load_episodes_stats(root: Path) -> dict[int, dict] | None:
    path = root / EPISODES_STATS_PATH
    if not path.exists():
        return None
    episodes_stats = load_jsonlines(path)
    episodes_stats = sorted(episodes_stats, key=lambda x: x["episode_index"])
    return {
        int(item["episode_index"]): cast_stats_to_numpy(item["stats"])
        for item in episodes_stats
    }


def load_stats(root: Path) -> dict | None:
    path = root / STATS_PATH
    if not path.exists():
        return None
    stats = load_json(path)
    return cast_stats_to_numpy(stats)


def get_episode_data_index_np(
    episode_dicts: dict[int, dict], episodes_subset: Iterable[int] | None = None
) -> dict[str, np.ndarray]:
    if episodes_subset is None:
        ordered = sorted(episode_dicts.keys())
    else:
        ordered = list(episodes_subset)
    lengths = [episode_dicts[ep_idx]["length"] for ep_idx in ordered]
    if not lengths:
        return {"from": np.array([], dtype=np.int64), "to": np.array([], dtype=np.int64)}
    cumulative = np.cumsum(lengths, dtype=np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    if len(lengths) > 1:
        starts[1:] = cumulative[:-1]
    return {"from": starts, "to": cumulative}


def get_episode_chunk(ep_idx: int, chunks_size: int) -> int:
    return int(ep_idx) // int(chunks_size)


def resolve_parquet_path(root: Path, info: dict, ep_idx: int) -> Path:
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    ep_chunk = get_episode_chunk(ep_idx, chunks_size)
    rel_path = info["data_path"].format(episode_chunk=ep_chunk, episode_index=ep_idx)
    return root / rel_path


def resolve_video_path(root: Path, info: dict, ep_idx: int, video_key: str) -> Path:
    if info.get("video_path") is None:
        raise ValueError("info.json does not define video_path for video features.")
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    ep_chunk = get_episode_chunk(ep_idx, chunks_size)
    rel_path = info["video_path"].format(
        episode_chunk=ep_chunk, video_key=video_key, episode_index=ep_idx
    )
    return root / rel_path


def decode_video_all_frames(
    video_path: Path,
    step: int = 1,
    max_frames: int | None = None,
    thread_count: int | None = None,
    thread_type: str | None = None,
) -> np.ndarray:
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    container = av.open(str(video_path))
    if thread_count is not None or thread_type is not None:
        stream = container.streams.video[0]
        if thread_type is not None:
            try:
                stream.thread_type = str(thread_type).upper()
            except (ValueError, AttributeError):
                pass
        if thread_count is not None:
            try:
                stream.thread_count = int(thread_count)
            except (ValueError, AttributeError):
                pass
    frames = []
    raw_count = 0
    try:
        for frame in container.decode(video=0):
            if max_frames is not None and raw_count >= max_frames:
                break
            if raw_count % step == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            raw_count += 1
        if max_frames is not None and raw_count < max_frames:
            raise ValueError(
                f"Frame count mismatch for {video_path}: {raw_count} < {max_frames}"
            )
    finally:
        container.close()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _decode_image_value(value, dataset_root: Path) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return np.array(Image.open(io.BytesIO(value["bytes"])).convert("RGB"), dtype=np.uint8)
        if value.get("path"):
            path = Path(value["path"])
            if not path.is_absolute():
                path = dataset_root / path
            return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        if "array" in value:
            return _normalize_image_array(np.array(value["array"]))
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute():
            path = dataset_root / path
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    if isinstance(value, np.ndarray):
        return _normalize_image_array(value)
    raise TypeError(f"Unsupported image value type: {type(value)}")


def load_image_sequence(values: list, dataset_root: Path) -> np.ndarray:
    frames = [_decode_image_value(v, dataset_root) for v in values]
    if not frames:
        raise ValueError("No image frames loaded.")
    return np.stack(frames, axis=0)


def resize_frames(frames: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    target_h, target_w = image_size
    resized = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        resized.append(np.array(img, dtype=np.uint8))
    return np.stack(resized, axis=0)


def _to_numpy(values: list, dtype: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not values:
        return np.zeros((0,) + tuple(expected_shape), dtype=DTYPE_MAP.get(dtype, np.float32))
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        arr = np.stack([np.asarray(v) for v in values], axis=0)
    else:
        arr = np.asarray(values)
    if dtype in DTYPE_MAP:
        arr = arr.astype(DTYPE_MAP[dtype], copy=False)
    if tuple(expected_shape) == (1,) and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _build_local_dataset(root: Path) -> dict:
    if not (root / INFO_PATH).exists():
        raise FileNotFoundError(f"info.json not found: {root / INFO_PATH}")
    info = load_info(root)
    features = info["features"]
    tasks, task_to_task_index = load_tasks(root)
    episodes = load_episodes(root)
    episodes_stats = load_episodes_stats(root)
    stats = load_stats(root)
    if episodes_stats is None and stats is not None:
        episodes_stats = {ep_idx: stats for ep_idx in episodes}
    if episodes_stats is None:
        episodes_stats = {}
    num_episodes = len(episodes)
    num_frames = sum(ep["length"] for ep in episodes.values())
    episode_data_index = get_episode_data_index_np(episodes)
    return {
        "root": root,
        "info": info,
        "features": features,
        "tasks": tasks,
        "task_to_task_index": task_to_task_index,
        "episodes": episodes,
        "episodes_stats": episodes_stats,
        "num_episodes": num_episodes,
        "num_frames": num_frames,
        "episode_data_index": episode_data_index,
        "repo_id": root.name,
    }


def _resolve_video_path_from_values(
    values: list | None, dataset_root: Path, fallback: Path
) -> Path:
    path = None
    if values and isinstance(values[0], dict) and values[0].get("path"):
        path = Path(values[0]["path"])
    elif values and isinstance(values[0], (str, Path)):
        path = Path(values[0])
    if path is None:
        path = fallback
    if not path.is_absolute():
        path = dataset_root / path
    return path


def _convert_episode(
    dataset: dict,
    ep_idx: int,
    features: dict,
    tasks_reversed: dict[str, int],
    image_size: tuple[int, int] | None,
    output_episode_index: int,
    fps_ratio: float | None = None,
    target_fps: int | None = None,
    video_decode_threads: int | None = None,
    video_decode_thread_type: str | None = None,
) -> dict[str, np.ndarray]:
    root = dataset["root"]
    info = dataset["info"]
    tasks = dataset["tasks"]

    parquet_path = resolve_parquet_path(root, info, ep_idx)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    available_columns = set(parquet_file.schema_arrow.names)
    columns_to_read = [k for k in features if k in available_columns]
    table = pq.read_table(
        parquet_path,
        columns=columns_to_read,
        use_threads=True,
        memory_map=True,
    )
    num_rows = table.num_rows
    columns = table.to_pydict()

    # 统一使用基于比率的采样：计算目标帧数和帧索引
    if fps_ratio is not None:
        expected_rows = int(num_rows / fps_ratio)
        if expected_rows == 0 and num_rows > 0:
            # 降采样后过短的 episode 至少保留第一帧，避免 0 帧导致下游出错
            expected_rows = 1
            frame_indices = np.array([0], dtype=np.int64)
        else:
            # 使用linspace生成均匀分布的帧索引，确保覆盖整个视频
            frame_indices = np.round(np.linspace(0, num_rows - 1, expected_rows)).astype(np.int64)
            expected_rows = len(frame_indices)
    else:
        expected_rows = num_rows
        frame_indices = None

    batch: dict[str, np.ndarray] = {}
    for key, ft in features.items():
        dtype = ft["dtype"]
        expected_shape = ft.get("shape", ())
        if dtype in ("image", "video"):
            values = columns.get(key)
            if dtype == "video":
                fallback_path = resolve_video_path(root, info, ep_idx, key)
                video_path = _resolve_video_path_from_values(values, root, fallback_path)
                if frame_indices is not None:
                    # 统一方法：先解码所有帧，然后使用索引采样
                    all_frames = decode_video_all_frames(
                        video_path,
                        step=1,
                        max_frames=num_rows,
                        thread_count=video_decode_threads,
                        thread_type=video_decode_thread_type,
                    )
                    frames = all_frames[frame_indices]
                else:
                    frames = decode_video_all_frames(
                        video_path,
                        step=1,
                        max_frames=num_rows,
                        thread_count=video_decode_threads,
                        thread_type=video_decode_thread_type,
                    )
            else:
                if values is None:
                    raise KeyError(f"Image feature '{key}' not found in parquet columns.")
                if len(values) < num_rows:
                    raise ValueError(
                        f"Frame count mismatch for {key} in episode {ep_idx}: "
                        f"{len(values)} < {num_rows}"
                    )
                if len(values) > num_rows:
                    values = values[:num_rows]
                all_frames = load_image_sequence(values, root)
                if frame_indices is not None:
                    frames = all_frames[frame_indices]
                else:
                    frames = all_frames
            if image_size is not None:
                frames = resize_frames(frames, image_size)
            batch[key] = frames
        else:
            if key not in columns:
                raise KeyError(f"Feature '{key}' not found in parquet columns.")
            arr = _to_numpy(columns[key], dtype, expected_shape)
            if arr.shape[0] != num_rows:
                raise ValueError(
                    f"Row count mismatch for {key} in episode {ep_idx}: "
                    f"{arr.shape[0]} != {num_rows}"
                )
            if frame_indices is not None:
                arr = arr[frame_indices]
            batch[key] = arr

    if fps_ratio is not None:
        if target_fps is None or target_fps <= 0:
            raise ValueError(f"target_fps must be positive when downsampling, got {target_fps}")
        for key in ("frame_index", "index", "timestamp"):
            if key not in batch:
                continue
            shape = tuple(features.get(key, {}).get("shape", ()))
            if shape == (1,):
                out_shape = (expected_rows, 1)
            elif shape:
                out_shape = (expected_rows,) + shape
            else:
                out_shape = (expected_rows,)
            dtype_name = features.get(key, {}).get("dtype", "")
            if key == "timestamp":
                dtype = DTYPE_MAP.get(dtype_name, np.float32)
                ts = (np.arange(expected_rows, dtype=np.float32) / float(target_fps)).astype(dtype)
                ts = ts.reshape(out_shape)
                batch[key] = ts
            else:
                dtype = DTYPE_MAP.get(dtype_name, np.int64)
                idx = np.arange(expected_rows, dtype=dtype).reshape(out_shape)
                batch[key] = idx

    for key, arr in batch.items():
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"Row count mismatch after subsampling for {key} in episode {ep_idx}: "
                f"{arr.shape[0]} != {expected_rows}"
            )

    num_rows = expected_rows

    if "episode_index" in batch:
        ep_shape = tuple(features["episode_index"].get("shape", (1,)))
        if ep_shape == (1,):
            batch["episode_index"] = np.full((num_rows, 1), output_episode_index, dtype=np.int64)
        else:
            batch["episode_index"] = np.full((num_rows,) + ep_shape, output_episode_index, dtype=np.int64)

    if "task_index" in batch and tasks_reversed:
        task_arr = batch["task_index"].reshape(-1)
        task_strings = [tasks[int(idx)] for idx in task_arr]
        new_task = np.array([tasks_reversed[t] for t in task_strings], dtype=np.int64)
        if tuple(features["task_index"].get("shape", (1,))) == (1,):
            new_task = new_task.reshape(-1, 1)
        batch["task_index"] = new_task

    return batch


def create_av_aloha_dataset_from_lerobot(
    dataset_root: str | Path | None = None,
    dataset_roots: list[str | Path] | None = None,
    root: str | Path | None = None,
    image_size: tuple[int, int] | None = None,
    remove_keys: list[str] = [],
    target_fps: int | None = None,
    compressors: str = "disk",
    video_decode_threads: int | None = None,
    video_decode_thread_type: str | None = None,
):
    if dataset_roots is None:
        if dataset_root is None:
            raise ValueError("Either dataset_root or dataset_roots must be provided.")
        dataset_roots = [dataset_root]

    dataset_roots = [Path(p).resolve() for p in dataset_roots]
    for p in dataset_roots:
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {p}")

    datasets = [_build_local_dataset(p) for p in dataset_roots]

    disabled_features = set()
    intersection_features = set(datasets[0]["features"])
    for ds in datasets:
        intersection_features.intersection_update(ds["features"])
    if len(intersection_features) == 0:
        raise RuntimeError(
            "Multiple datasets were provided but they had no keys common to all of them. "
            "The multi-dataset functionality currently only keeps common keys."
        )
    for ds in datasets:
        extra_keys = set(ds["features"]).difference(intersection_features)
        if extra_keys:
            logger.info(
                f"keys {extra_keys} of {ds['repo_id']} were disabled as they are not contained in all the "
                "other datasets."
            )
        disabled_features.update(extra_keys)
    logger.info(f"Disabled features: {disabled_features}.\n")

    source_fps = datasets[0]["info"]["fps"]
    if not all(ds["info"]["fps"] == source_fps for ds in datasets):
        raise AssertionError("Datasets have different fps values.")

    fps = source_fps
    fps_ratio = None  # source_fps / target_fps，用于统一采样
    if target_fps is not None:
        if fps < target_fps:
             logger.warning(f"Target FPS {target_fps} is higher than source FPS {fps}. Ignoring target FPS.")
        elif fps == target_fps:
            pass
        else:
            fps_ratio = fps / target_fps
            logger.info(f"Downsampling from {fps} FPS to {target_fps} FPS (ratio={fps_ratio:.3f})")
            fps = target_fps

    if fps_ratio is not None:
        num_frames = 0
        for d in datasets:
             for ep in d["episodes"].values():
                  length = ep["length"]
                  num_frames += int(length / fps_ratio)
    else:
        num_frames = sum(d["num_frames"] for d in datasets)

    num_episodes = sum(d["num_episodes"] for d in datasets)

    features = {}
    for dataset in datasets:
        features.update({k: v for k, v in dataset["features"].items()})
    features = {k: v for k, v in features.items() if k not in disabled_features}
    features = {k: v for k, v in features.items() if k not in remove_keys}

    if fps_ratio is not None:
        for k, v in features.items():
            if v.get("dtype") in ("video", "image") and "info" in v:
                if "video.fps" in v["info"]:
                    v["info"]["video.fps"] = fps
                    logger.info(f"Updated video.fps for {k} to {fps}")

    if image_size is not None:
        target_h, target_w = image_size
        for k, v in features.items():
            if v.get("dtype") in ("video", "image"):
                shape = tuple(v.get("shape", ()))
                if len(shape) < 2:
                    logger.warning(
                        f"Cannot update shape for {k}: expected at least 2 dims, got {shape}"
                    )
                    continue
                v["shape"] = (target_h, target_w) + shape[2:]
                info = v.get("info")
                if isinstance(info, dict):
                    if "video.height" in info:
                        info["video.height"] = target_h
                    if "video.width" in info:
                        info["video.width"] = target_w
                logger.info(f"Updated shape for {k} to {v['shape']}")

    camera_keys = [k for k, ft in features.items() if ft["dtype"] in ("image", "video")]
    video_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
    image_keys = [k for k, ft in features.items() if ft["dtype"] == "image"]

    episodes_stats = []
    for dataset in datasets:
        if dataset["episodes_stats"]:
            for _, ep_stats in dataset["episodes_stats"].items():
                episodes_stats.append({k: v for k, v in ep_stats.items() if k in features})
    stats = aggregate_stats(episodes_stats) if episodes_stats else {}

    tasks = []
    for ds in datasets:
        tasks.extend(ds["tasks"].values())
    tasks = {i: task for i, task in enumerate(tasks)}
    tasks_reversed = {v: k for k, v in tasks.items()}

    if root is None:
        raise ValueError("Output root must be provided.")
    root = Path(root)

    if root.exists():
        logger.info(f"Removing existing directory {root}...")
        shutil.rmtree(root)

    replay_buffer = ReplayBuffer.create_from_path(zarr_path=root, mode="a")

    repo_id = datasets[0]["repo_id"] if len(datasets) == 1 else "+".join([d["repo_id"] for d in datasets])

    episode_idx = 0
    total_episodes_count = sum(len(dataset["episodes"]) for dataset in datasets)
    
    with tqdm(total=total_episodes_count, desc="Converting Episodes") as pbar:
        for dataset in datasets:
            ep_indices = sorted(dataset["episodes"].keys())
            for ep_idx in ep_indices:
                batch = _convert_episode(
                    dataset=dataset,
                    ep_idx=ep_idx,
                    features=features,
                    tasks_reversed=tasks_reversed,
                    image_size=image_size,
                    output_episode_index=episode_idx,
                    fps_ratio=fps_ratio,
                    target_fps=fps,
                    video_decode_threads=video_decode_threads,
                    video_decode_thread_type=video_decode_thread_type,
                )
                episode_length = next(iter(batch.values())).shape[0] if batch else 0
                replay_buffer.add_episode(batch, compressors=compressors)
                episode_idx += 1
                pbar.update(1)

    config = {
        "repo_id": repo_id,
        "stats": stats,
        "num_frames": replay_buffer.n_steps,
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
    logger.info(f"Converted dataset saved to {root}.")
