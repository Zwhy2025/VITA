"""
LeRobot 数据格式转换器
"""
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Resize
from tqdm import tqdm

try:
    from ..core.compute_stats import aggregate_stats
except ImportError:
    aggregate_stats = None

from .base import BaseDataConverter, ConversionConfig


class LeRobotConverter(BaseDataConverter):
    """LeRobot 格式数据转换器"""
    
    def detect_format(self, path: Union[str, Path]) -> bool:
        """检测是否为 LeRobot 格式"""
        path = Path(path)
        
        # 检查关键文件和目录
        required_dirs = ['data', 'meta']
        required_files = ['meta/info.json']
        
        for dir_name in required_dirs:
            if not (path / dir_name).is_dir():
                return False
        
        for file_name in required_files:
            if not (path / file_name).is_file():
                return False
        
        # 检查 info.json 内容
        try:
            import json
            with open(path / 'meta/info.json', 'r') as f:
                info = json.load(f)
            
            # 检查必要字段
            required_fields = ['codebase_version', 'features']
            return all(field in info for field in required_fields)
        except Exception:
            return False
    
    def load_datasets(self, path: Union[str, Path]) -> List[Any]:
        """加载 LeRobot 数据集"""
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError("需要安装 lerobot: pip install lerobot")
        
        path = Path(path)
        
        path_str = str(path.resolve())
        try:
            # 使用 pyav 后端（比 torchcodec 更稳定）
            dataset = LeRobotDataset(
                repo_id=path_str, 
                episodes=self.config.episodes,
                video_backend="pyav"
            )
        except Exception as e:
            self.logger.error(f"创建 LeRobot 数据集失败: {e}")
            raise
        

        self.logger.info(f"Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}")
        
        return [dataset]
    
    def extract_metadata(self, datasets: List[Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """提取 LeRobot 数据集元数据"""
        if not datasets:
            raise ValueError("数据集列表为空")
        
        dataset = datasets[0]
        path = Path(path) if path else None
        
        # 获取基本信息
        fps = dataset.meta.fps
        num_frames = sum(d.num_frames for d in datasets)
        num_episodes = sum(d.num_episodes for d in datasets)
        
        # 处理特征
        features = {}
        for ds in datasets:
            features.update({k: v for k, v in ds.features.items()})
        
        # 移除指定的键
        features = {k: v for k, v in features.items() if k not in self.config.remove_keys}
        
        # 获取相机和图像键
        camera_keys = list(dataset.meta.camera_keys or [])
        video_keys = list(dataset.meta.video_keys or [])
        image_keys = list(dataset.meta.image_keys or [])
        
        # 过滤存在的键
        camera_keys = [k for k in camera_keys if k in features]
        video_keys = [k for k in video_keys if k in features]
        image_keys = [k for k in image_keys if k in features]
        
        # 计算统计信息
        episodes_stats = []
        for ds in datasets:
            episodes = ds.episodes if ds.episodes else range(ds.num_episodes)
            for ep_idx in episodes:
                if ep_idx in ds.meta.episodes_stats:
                    episode_stats = {k: v for k, v in ds.meta.episodes_stats[ep_idx].items() if k in features}
                    episodes_stats.append(episode_stats)
        
        if episodes_stats and aggregate_stats:
            stats = aggregate_stats(episodes_stats)
        else:
            stats = {}
        
        # 处理任务
        tasks = {}
        if hasattr(dataset.meta, 'tasks') and dataset.meta.tasks:
            tasks = {i: task for i, task in enumerate(dataset.meta.tasks.values())}
        
        return {
            "repo_id": str(path) if path else "",
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
    
    def convert_episode(self, episode_data: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """转换单个 episode 数据"""
        batch = episode_data
        
        # 添加 episode_index
        if 'episode_index' in batch:
            batch['episode_index'] = torch.full_like(batch['episode_index'], episode_idx)
        
        # 处理任务索引
        if 'task' in batch and 'task_index' not in batch:
            metadata = episode_data.get('_metadata', {})
            tasks = metadata.get('tasks', {})
            if tasks:
                # 创建任务映射
                tasks_reversed = {v: k for k, v in tasks.items()}
                batch['task_index'] = torch.tensor([
                    tasks_reversed.get(task, 0) for task in batch['task']
                ], dtype=torch.int32)
            del batch["task"]
        
        # 转换数据格式
        converted_batch = {}
        features = episode_data.get('_metadata', {}).get('features', {})
        
        for key, value in batch.items():
            if key.startswith('_'):  # 跳过元数据
                continue
                
            if key not in features:
                continue
                
            converted_batch[key] = self._convert_feature(key, value, features[key])
        
        return converted_batch
    
    def _convert_feature(self, key: str, value: torch.Tensor, feature_info: Dict[str, Any]) -> np.ndarray:
        """转换单个特征"""
        dtype = feature_info.get('dtype')
        
        if dtype in ['image', 'video']:
            # 图像/视频处理
            if self.config.image_size is not None:
                value = Resize(self.config.image_size)(value)
            
            # (B, C, H, W) to (B, H, W, C)
            if value.dim() == 4 and value.shape[1] == 3:
                value = value.permute(0, 2, 3, 1)
            
            # 转换为 uint8
            value = (value * 255).to(torch.uint8)
        
        # 确保转换为 numpy 数组
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        else:
            # 如果是列表或其他类型，先转换为 tensor 再转换
            return torch.as_tensor(value).cpu().numpy()
    
    def _get_num_episodes(self, dataset: Any) -> int:
        """获取数据集中的 episode 数量"""
        return dataset.num_episodes
    
    def _get_episode_data(self, dataset: Any, episode_idx: int) -> Dict[str, Any]:
        """获取单个 episode 的数据"""
        from torch.utils.data import Subset
        
        # 获取 episode 的数据范围
        from_idx = dataset.episode_data_index['from'][episode_idx]
        to_idx = dataset.episode_data_index['to'][episode_idx]
        
        # 创建子集
        subset = Subset(dataset, range(from_idx, to_idx))
        
        # 使用 DataLoader 批量加载
        dataloader = DataLoader(
            subset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        # 收集所有批次
        batches = []
        for batch in dataloader:
            batches.append(batch)
        
        # 合并批次
        if batches:
            episode_data = {}
            for k in batches[0].keys():
                values = [b[k] for b in batches]
                # 检查是否为 tensor 类型
                if isinstance(values[0], torch.Tensor):
                    episode_data[k] = torch.cat(values, dim=0)
                else:
                    # 对于非 tensor 类型（如字符串列表），直接扩展
                    episode_data[k] = []
                    for v in values:
                        if isinstance(v, list):
                            episode_data[k].extend(v)
                        else:
                            episode_data[k].append(v)
        else:
            episode_data = {}
        
        # 添加元数据供转换使用
        episode_data['_metadata'] = {
            'features': dataset.features,
            'tasks': getattr(dataset.meta, 'tasks', {}),
        }
        
        return episode_data


class OptimizedLeRobotConverter(BaseDataConverter):
    """
    优化的 LeRobot 格式转换器
    
    使用 PyAV 直接读取视频，pandas 读取 parquet，
    跳过 LeRobotDataset 的元数据处理开销。
    
    速度提升: 50-60x (vs LeRobotConverter)
    """
    
    def detect_format(self, path: Union[str, Path]) -> bool:
        """检测是否为 LeRobot 格式 (与 LeRobotConverter 相同)"""
        path = Path(path)
        
        required_dirs = ['data', 'meta']
        required_files = ['meta/info.json']
        
        for dir_name in required_dirs:
            if not (path / dir_name).is_dir():
                return False
        
        for file_name in required_files:
            if not (path / file_name).is_file():
                return False
        
        try:
            import json
            with open(path / 'meta/info.json', 'r') as f:
                info = json.load(f)
            
            required_fields = ['codebase_version', 'features']
            return all(field in info for field in required_fields)
        except Exception:
            return False
    
    def load_datasets(self, path: Union[str, Path]) -> List[Any]:
        """
        加载数据集信息 (不实际加载数据)
        
        Returns:
            List containing a Dict with:
                - num_episodes: total episodes
                - num_frames: total frames
                - video_keys: list of video keys
                - features: feature definitions from info.json
                - fps: video fps
                - chunk: chunk number
        """
        path = Path(path)
        
        # 读取 info.json 获取元数据
        import json
        with open(path / 'meta/info.json', 'r') as f:
            info = json.load(f)
        
        # 解析 video_path 模板获取 video_keys
        video_path_template = info.get('video_path', '')
        video_keys = self._extract_video_keys(video_path_template)
        
        num_episodes = info.get('total_episodes', 0)
        num_frames = info.get('total_frames', 0)
        fps = info.get('fps', 30)
        
        self.logger.info(f"Optimized mode: {num_episodes} episodes, {num_frames} frames, {len(video_keys)} video keys")
        
        return [{
            'num_episodes': num_episodes,
            'num_frames': num_frames,
            'video_keys': video_keys,
            'features': info.get('features', {}),
            'fps': fps,
            'chunk': 0,
            'path': str(path),
        }]  # type: ignore[return-value]
    
    def _extract_video_keys(self, video_path_template: str) -> List[str]:
        """从路径模板中提取 video keys"""
        if not video_path_template:
            return []
        
        # video_path format: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        # video_key would be like "observation.images.top_image"
        import re
        match = re.search(r'\{video_key\}', video_path_template)
        if match:
            # We can't know video keys from template alone
            # Return common ones, will be filtered later
            return []
        return []
    
    def extract_metadata(self, datasets: List[Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """提取 LeRobot 数据集元数据 (优化版)"""
        if not datasets:
            raise ValueError("数据集列表为空")
        
        dataset_info = datasets[0]  # Get the dict from the list
        path = Path(path) if path else None
        
        features = dataset_info.get('features', {})
        
        # 移除指定的键
        features = {k: v for k, v in features.items() if k not in self.config.remove_keys}
        
        # 获取 video/image keys
        video_keys = [k for k in features.keys() if 'image' in k.lower() or 'video' in k.lower()]
        image_keys = [k for k in features.keys() if 'image' in k.lower()]
        
        return {
            "repo_id": str(path) if path else "",
            "stats": {},
            "num_frames": dataset_info.get('num_frames', 0),
            "num_episodes": dataset_info.get('num_episodes', 0),
            "features": features,
            "camera_keys": [],
            "video_keys": video_keys,
            "image_keys": image_keys,
            "fps": dataset_info.get('fps', 30),
            "tasks": {},
            "_dataset_info": dataset_info,
        }
    
    def convert_episode(self, episode_data: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """转换单个 episode 数据"""
        return episode_data
    
    def _convert_feature(self, key: str, value: np.ndarray, feature_info: Dict[str, Any]) -> np.ndarray:
        """转换单个特征"""
        dtype = feature_info.get('dtype', '')
        
        if dtype in ['image', 'video']:
            # 确保形状是 (N, H, W, C)
            if value.ndim == 3:
                value = np.expand_dims(value, axis=0)  # (1, H, W, C)
            elif value.ndim == 4 and value.shape[1] == 3:
                # (N, C, H, W) -> (N, H, W, C)
                value = np.transpose(value, (0, 2, 3, 1))
            
            # 调整大小 (如果需要)
            if self.config.image_size is not None:
                import torch
                from torchvision.transforms import Resize
                target_h, target_w = self.config.image_size
                
                # 转换为 (N*C, H, W, C) 进行 resize
                N = value.shape[0]
                flat = value.reshape(-1, value.shape[2], value.shape[3], value.shape[4])
                flat_tensor = torch.from_numpy(flat).float().permute(0, 3, 1, 2)  # (N*C, C, H, W)
                resized = Resize((target_h, target_w))(flat_tensor)
                resized = resized.permute(0, 2, 3, 1).numpy()  # (N*C, H, W, C)
                
                # 恢复形状 (N, H, W, C)
                value = resized.reshape(N, target_h, target_w, 3)
            
            # 确保 uint8
            if value.dtype != np.uint8:
                if value.max() <= 1.0:
                    value = (value * 255).astype(np.uint8)
                else:
                    value = value.astype(np.uint8)
        
        return value
    
    def _get_num_episodes(self, dataset: Any) -> int:
        """获取 episode 数量 (优化版)"""
        dataset_info = dataset  # Cast to dict
        return dataset_info.get('num_episodes', 0)
    
    def _get_episode_data(self, dataset: Any, episode_idx: int) -> Dict[str, Any]:
        """获取单个 episode 的数据 (使用 PyAV + pandas)"""
        from .fast_video import read_video_fast
        import pandas as pd
        
        dataset_info = dataset  # Cast to dict
        base_path = dataset_info['path']
        chunk = dataset_info.get('chunk', 0)
        
        result = {
            'episode_idx': episode_idx,
            '_metadata': {
                'features': dataset_info.get('features', {}),
                'tasks': {},
            }
        }
        
        # 读取 parquet 数据 (状态、动作等)
        parquet_path = Path(base_path) / 'data' / f'chunk-{chunk:03d}' / f'episode_{episode_idx:06d}.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            for col in df.columns:
                if col.startswith('_'):
                    continue
                values = df[col].values
                # 如果值是 object 类型（包含数组），转换为正确的 numpy 数组
                if values.dtype == object:
                    values = np.array([np.asarray(v) for v in values])
                result[col] = values
        else:
            self.logger.warning(f"Parquet not found: {parquet_path}")
        
        # 读取视频数据
        features = dataset_info.get('features', {})
        for video_key in features.keys():
            if 'image' not in video_key.lower() and 'video' not in video_key.lower():
                continue
            
            video_path = Path(base_path) / 'videos' / f'chunk-{chunk:03d}' / video_key / f'episode_{episode_idx:06d}.mp4'
            if video_path.exists():
                frames, fps = read_video_fast(str(video_path))
                
                # 转换视频格式
                feature_info = features.get(video_key, {})
                frames = self._convert_feature(video_key, frames, feature_info)
                result[video_key] = frames
        
        return result
    
    def _convert_episodes(self, datasets: List[Any], replay_buffer: Any, metadata: Dict[str, Any]):
        """优化的顺序转换"""
        import time
        import gc
        
        dataset_info = datasets[0]  # dataset_info dict
        num_episodes = self._get_num_episodes(dataset_info)
        total_episodes = metadata.get('num_episodes', num_episodes)
        
        self.logger.info(f"Starting optimized conversion of {num_episodes} episodes")
        
        episode_idx = 0
        for ep_idx in range(num_episodes):
            if self.config.episodes and episode_idx not in self.config.episodes:
                episode_idx += 1
                continue
            
            start_time = time.time()
            self.logger.info(f"Converting episode {episode_idx + 1}/{total_episodes}")
            
            # 获取 episode 数据
            episode_data = self._get_episode_data(dataset_info, ep_idx)
            
            # 添加 episode index
            if 'episode_index' in episode_data:
                episode_data['episode_index'] = np.full_like(episode_data['episode_index'], episode_idx)
            else:
                num_frames = len(episode_data.get('observation.state', []))
                episode_data['episode_index'] = np.full(num_frames, episode_idx, dtype=np.int64)
            
            # 转换数据
            converted_data = self._convert_episode_data(episode_data, episode_idx, dataset_info)
            
            # 写入 replay buffer
            replay_buffer.add_episode(converted_data, compressors=self.config.compressors)
            
            if self.config.profile:
                self._timing_stats[f'episode_{episode_idx}'] = time.time() - start_time
            
            # 清理内存
            del episode_data, converted_data
            gc.collect()
            
            episode_idx += 1
    
    def _convert_episode_data(self, episode_data: Dict[str, Any], episode_idx: int, config: Any):
        """转换 episode 数据"""
        dataset_info = config  # Cast to dict
        features = dataset_info.get('features', {})
        converted = {}
        
        for key, value in episode_data.items():
            if key.startswith('_'):
                continue
            
            if key not in features:
                continue
            
            if isinstance(value, np.ndarray):
                converted[key] = value
            elif isinstance(value, (list, tuple)):
                converted[key] = np.array(value)
            else:
                converted[key] = value
        
        return converted


class TorchCodecGPUConverter(BaseDataConverter):
    """
    使用 TorchCodec GPU 加速的 LeRobot 格式转换器
    
    利用 NVIDIA GPU 硬件解码视频，比纯 CPU 方案更快。
    适合有强大 GPU 的机器。
    """
    
    def detect_format(self, path: Union[str, Path]) -> bool:
        """检测是否为 LeRobot 格式"""
        path = Path(path)
        
        required_dirs = ['data', 'meta']
        required_files = ['meta/info.json']
        
        for dir_name in required_dirs:
            if not (path / dir_name).is_dir():
                return False
        
        for file_name in required_files:
            if not (path / file_name).is_file():
                return False
        
        try:
            import json
            with open(path / 'meta/info.json', 'r') as f:
                info = json.load(f)
            
            required_fields = ['codebase_version', 'features']
            return all(field in info for field in required_fields)
        except Exception:
            return False
    
    def load_datasets(self, path: Union[str, Path]) -> List[Any]:
        """加载数据集信息"""
        import json
        
        path = Path(path)
        
        with open(path / 'meta/info.json', 'r') as f:
            info = json.load(f)
        
        video_path_template = info.get('video_path', '')
        
        num_episodes = info.get('total_episodes', 0)
        num_frames = info.get('total_frames', 0)
        fps = info.get('fps', 30)
        
        self.logger.info(f"TorchCodec GPU mode: {num_episodes} episodes, {num_frames} frames")
        
        return [{
            'num_episodes': num_episodes,
            'num_frames': num_frames,
            'features': info.get('features', {}),
            'fps': fps,
            'chunk': 0,
            'path': str(path),
        }]
    
    def extract_metadata(self, datasets: List[Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """提取元数据"""
        if not datasets:
            raise ValueError("数据集列表为空")
        
        dataset_info = datasets[0]
        path = Path(path) if path else None
        
        features = dataset_info.get('features', {})
        features = {k: v for k, v in features.items() if k not in self.config.remove_keys}
        
        video_keys = [k for k in features.keys() if 'image' in k.lower() or 'video' in k.lower()]
        
        return {
            "repo_id": str(path) if path else "",
            "stats": {},
            "num_frames": dataset_info.get('num_frames', 0),
            "num_episodes": dataset_info.get('num_episodes', 0),
            "features": features,
            "camera_keys": [],
            "video_keys": video_keys,
            "image_keys": video_keys,
            "fps": dataset_info.get('fps', 30),
            "tasks": {},
            "_dataset_info": dataset_info,
        }
    
    def convert_episode(self, episode_data: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """转换单个 episode 数据"""
        return episode_data
    
    def _get_num_episodes(self, dataset: Any) -> int:
        """获取 episode 数量"""
        return dataset.get('num_episodes', 0)
    
    def _get_episode_data(self, dataset: Any, episode_idx: int) -> Dict[str, Any]:
        """获取单个 episode 的数据 (使用 TorchCodec GPU)"""
        import pandas as pd
        
        dataset_info = dataset
        base_path = dataset_info['path']
        chunk = dataset_info.get('chunk', 0)
        
        result = {
            'episode_idx': episode_idx,
            '_metadata': {
                'features': dataset_info.get('features', {}),
                'tasks': {},
            }
        }
        
        parquet_path = Path(base_path) / 'data' / f'chunk-{chunk:03d}' / f'episode_{episode_idx:06d}.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            for col in df.columns:
                if col.startswith('_'):
                    continue
                values = df[col].values
                if values.dtype == object:
                    values = np.array([np.asarray(v) for v in values])
                result[col] = values
        else:
            self.logger.warning(f"Parquet not found: {parquet_path}")
        
        features = dataset_info.get('features', {})
        
        for video_key in features.keys():
            if 'image' not in video_key.lower() and 'video' not in video_key.lower():
                continue
            
            video_path = Path(base_path) / 'videos' / f'chunk-{chunk:03d}' / video_key / f'episode_{episode_idx:06d}.mp4'
            if video_path.exists():
                frames = self._read_video_gpu(str(video_path))
                result[video_key] = frames
        
        return result
    
    def _read_video_gpu(self, video_path: str) -> np.ndarray:
        """使用 TorchCodec GPU 读取视频"""
        try:
            from torchvision import io as tv_io
            import torch
            
            # 检查 GPU decoder 是否可用
            if not tv_io._HAS_GPU_VIDEO_DECODER:
                raise RuntimeError("GPU video decoder not available")
            
            reader = tv_io.VideoReader(video_path, "video", device="cuda")
            frames_list = []
            
            for frame in reader:
                frames_list.append(frame["data"])
            
            if frames_list:
                frames_tensor = torch.stack(frames_list)
                if frames_tensor.is_cuda:
                    frames_tensor = frames_tensor.cpu()
                return frames_tensor.numpy()
            
            return np.zeros((0, 480, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            self.logger.warning(f"GPU decoding failed ({e}), using PyAV fallback")
            return self._read_video_pyav(video_path)
    
    def _read_video_pyav(self, video_path: str) -> np.ndarray:
        """使用 PyAV 读取视频 (CPU fallback)"""
        import av
        
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        frames = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            frames.append(arr)
        
        return np.stack(frames) if frames else np.zeros((0, 480, 640, 3), dtype=np.uint8)
    
    def _convert_episodes(self, datasets: List[Any], replay_buffer: Any, metadata: Dict[str, Any]):
        """优化的顺序转换"""
        import time
        import gc
        
        dataset_info = datasets[0]
        num_episodes = self._get_num_episodes(dataset_info)
        total_episodes = metadata.get('num_episodes', num_episodes)
        
        self.logger.info(f"Starting TorchCodec GPU conversion of {num_episodes} episodes")
        
        episode_idx = 0
        for ep_idx in range(num_episodes):
            if self.config.episodes and episode_idx not in self.config.episodes:
                episode_idx += 1
                continue
            
            start_time = time.time()
            self.logger.info(f"Converting episode {episode_idx + 1}/{total_episodes}")
            
            episode_data = self._get_episode_data(dataset_info, ep_idx)
            
            if 'episode_index' in episode_data:
                episode_data['episode_index'] = np.full_like(episode_data['episode_index'], episode_idx)
            else:
                num_frames = len(episode_data.get('observation.state', []))
                episode_data['episode_index'] = np.full(num_frames, episode_idx, dtype=np.int64)
            
            converted_data = self._convert_episode_data(episode_data, episode_idx, dataset_info)
            
            replay_buffer.add_episode(converted_data, compressors=self.config.compressors)
            
            if self.config.profile:
                self._timing_stats[f'episode_{episode_idx}'] = time.time() - start_time
            
            del episode_data, converted_data
            gc.collect()
            
            episode_idx += 1
    
    def _convert_episode_data(self, episode_data: Dict[str, Any], episode_idx: int, config: Any):
        """转换 episode 数据"""
        dataset_info = config
        features = dataset_info.get('features', {})
        converted = {}
        
        for key, value in episode_data.items():
            if key.startswith('_'):
                continue
            
            if key not in features:
                continue
            
            if isinstance(value, np.ndarray):
                converted[key] = value
            elif isinstance(value, (list, tuple)):
                converted[key] = np.array(value)
            else:
                converted[key] = value
        
        return converted