"""
数据转换器抽象基类
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import shutil
import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

import torch
import numpy as np
from torchvision.transforms import Resize

from ..core.replay_buffer import ReplayBuffer
from ..core.compute_stats import aggregate_stats


@dataclass
class ConversionConfig:
    """数据转换配置"""
    output_root: Optional[Union[str, Path]] = None
    episodes: Optional[List[int]] = None
    remove_keys: Optional[List[str]] = None
    image_size: Optional[tuple] = None
    compressors: str = 'disk'
    batch_size: int = 16
    num_workers: int = 1
    parallel: bool = False
    profile: bool = True  # 默认开启性能分析
    
    def __post_init__(self):
        if self.remove_keys is None:
            self.remove_keys = []


class BaseDataConverter(ABC):
    """数据转换器抽象基类"""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._timing_stats = {}
        
    @abstractmethod
    def detect_format(self, path: Union[str, Path]) -> bool:
        """检测是否支持该数据格式"""
        pass
    
    @abstractmethod
    def load_datasets(self, path: Union[str, Path]) -> List[Any]:
        """加载数据集"""
        pass
    
    @abstractmethod
    def extract_metadata(self, datasets: List[Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """提取元数据"""
        pass
    
    @abstractmethod
    def convert_episode(self, episode_data: Any, episode_idx: int) -> Dict[str, Any]:
        """转换单个episode数据"""
        pass
    
    def convert(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """执行数据转换"""
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = self._get_default_output_path(input_path)
        output_path = Path(output_path)
        
        self.logger.info(f"开始转换数据集: {input_path} -> {output_path}")
        
        if not self.detect_format(input_path):
            raise ValueError(f"不支持的格式: {input_path}")
        
        datasets = self.load_datasets(input_path)
        self.logger.info(f"加载了 {len(datasets)} 个数据集")
        
        metadata = self.extract_metadata(datasets, input_path)
        
        # 创建输出目录
        output_path_str = str(output_path)
        if output_path.exists():
            self.logger.info(f"删除现有目录: {output_path}")
            import os as os_mod
            for root_dir, dirs, files in os_mod.walk(output_path):
                for file in files:
                    file_path = os_mod.path.join(root_dir, file)
                    try:
                        os_mod.remove(file_path)
                    except Exception:
                        pass
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 replay buffer
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=output_path_str, mode="w")
        
        # 保存配置
        self._save_config(output_path, metadata)
        
        # 转换episodes
        if self.config.parallel and self.config.num_workers > 1:
            self._convert_episodes_parallel(datasets, replay_buffer, metadata, str(input_path))
        else:
            self._convert_episodes(datasets, replay_buffer, metadata)
        
        self.logger.info(f"转换完成: {output_path}")
        
        if self.config.profile:
            self._print_timing_stats()
        
        return output_path
    
    def _print_timing_stats(self):
        """打印性能分析结果"""
        self.logger.info("=" * 50)
        self.logger.info("性能分析结果:")
        self.logger.info("=" * 50)
        total = sum(self._timing_stats.values())
        for name, duration in sorted(self._timing_stats.items(), key=lambda x: -x[1]):
            pct = 100 * duration / total if total > 0 else 0
            self.logger.info(f"  {name}: {duration:.1f}s ({pct:.1f}%)")
        self.logger.info(f"  总计: {total:.1f}s")
        self.logger.info("=" * 50)
    
    def _get_default_output_path(self, input_path: Path) -> Path:
        """获取默认输出路径"""
        if 'FLARE_DATASETS_DIR' in os.environ:
            base_dir = Path(os.environ['FLARE_DATASETS_DIR'])
        else:
            base_dir = Path(__file__).parent.parent.parent / "gym-av-aloha" / "outputs"
        
        safe_name = input_path.name.replace("/", "_").replace("\\", "_")
        return base_dir / safe_name
    
    def _save_config(self, output_path: Path, metadata: Dict[str, Any]):
        """保存配置文件"""
        import json
        
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._make_json_serializable(metadata), f, indent=4)
        
        self.logger.info(f"配置已保存: {config_path}")
    
    def _convert_episodes(self, datasets: List[Any], replay_buffer: ReplayBuffer, metadata: Dict[str, Any]):
        """顺序转换所有episodes"""
        episode_idx = 0
        total_episodes = metadata.get('num_episodes', 0)
        
        for dataset_idx, dataset in enumerate(datasets):
            num_episodes = self._get_num_episodes(dataset)
            
            for ep_idx in range(num_episodes):
                if self.config.episodes and episode_idx not in self.config.episodes:
                    episode_idx += 1
                    continue
                
                start_time = time.time()
                self.logger.info(f"转换 episode {episode_idx + 1}/{total_episodes} (数据集 {dataset_idx + 1}, episode {ep_idx + 1})")
                
                episode_data = self._get_episode_data(dataset, ep_idx)
                converted_data = self.convert_episode(episode_data, episode_idx)
                
                replay_buffer.add_episode(converted_data, compressors=self.config.compressors)
                
                if self.config.profile:
                    self._timing_stats[f'episode_{episode_idx}'] = time.time() - start_time
                
                episode_idx += 1
    
    def _convert_episodes_parallel(self, datasets: List[Any], replay_buffer: ReplayBuffer, metadata: Dict[str, Any], input_path: str):
        """并行转换 - 警告：可能导致 OOM（每个 episode 需要 ~11GB 内存）"""
        from tqdm import tqdm
        
        self.logger.warning("警告：并行转换可能导致 OOM。每个 episode 转换需要 ~11GB 内存。")
        self.logger.warning("建议使用顺序转换（不加 --parallel）以获得稳定性能。")
        
        # 保守限制 workers：只允许 1 个 worker（等同于顺序转换，但有进程开销）
        num_workers = 1
        self.logger.info(f"使用 {num_workers} 个工作进程（限制以避免 OOM）")
        
        # 收集所有需要转换的 episode 任务
        tasks = []
        episode_idx = 0
        total_episodes = metadata.get('num_episodes', 0)
        
        for dataset_idx, dataset in enumerate(datasets):
            num_episodes = self._get_num_episodes(dataset)
            
            for ep_idx in range(num_episodes):
                if self.config.episodes and episode_idx not in self.config.episodes:
                    episode_idx += 1
                    continue
                
                # 只收集 from/to 索引，不加载数据
                from_idx = dataset.episode_data_index['from'][ep_idx]
                to_idx = dataset.episode_data_index['to'][ep_idx]
                
                tasks.append({
                    'episode_idx': episode_idx,
                    'from_idx': from_idx,
                    'to_idx': to_idx,
                })
                episode_idx += 1
        
        self.logger.info(f"准备转换 {len(tasks)} 个 episodes")
        
        # 并行执行
        completed = 0
        start_time = time.time()
        
        # 序列化 zarr 写入
        zarr_lock = multiprocessing.Lock()
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = []
            
            for task in tasks:
                future = executor.submit(
                    self._convert_single_episode_worker,
                    task,
                    input_path,
                    self.config
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="转换 episodes"):
                try:
                    result = future.result()
                    completed += 1
                    
                    if result.get('success') and 'data' in result:
                        with zarr_lock:
                            replay_buffer.add_episode(result['data'], compressors=self.config.compressors)
                        
                        if self.config.profile:
                            self._timing_stats[f'episode_{result["episode_idx"]}'] = result.get('duration', 0)
                            
                    elif result.get('error'):
                        self.logger.warning(f"Episode {result.get('episode_idx')} 出错: {result['error']}")
                        
                except Exception as e:
                    self.logger.error(f"处理错误: {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"并行转换完成: {completed}/{len(tasks)} 个 episodes, 用时 {elapsed:.1f}s")
    
    def _convert_single_episode_worker(self, task: Dict, input_path: str, config: 'ConversionConfig') -> Dict:
        """Worker 函数 - 只加载需要的 episode 数据"""
        import tracemalloc
        
        try:
            import sys
            from pathlib import Path
            import numpy as np
            import torch
            from torch.utils.data import DataLoader, Subset
            import gc
            import time
            import psutil
            
            tracemalloc.start()
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 添加项目路径
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Step 1: 加载数据集
            step1_start = time.time()
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(repo_id=str(Path(input_path).resolve()))
            step1_time = time.time() - step1_start
            step1_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Step 2: 创建 subset 和 dataloader
            step2_start = time.time()
            from_idx = task['from_idx']
            to_idx = task['to_idx']
            subset = Subset(dataset, range(from_idx, to_idx))
            dataloader = DataLoader(subset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            step2_time = time.time() - step2_start
            
            # Step 3: 收集批次
            step3_start = time.time()
            batches = []
            for batch in dataloader:
                batches.append(batch)
                if len(batches) > 10:
                    del batches[0]
                    gc.collect()
            step3_time = time.time() - step3_start
            step3_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Step 4: 合并批次
            step4_start = time.time()
            episode_data = {}
            if batches:
                for k in batches[0].keys():
                    values = [b[k] for b in batches]
                    if isinstance(values[0], torch.Tensor):
                        episode_data[k] = torch.cat(values, dim=0)
                    else:
                        episode_data[k] = []
                        for v in values:
                            if isinstance(v, list):
                                episode_data[k].extend(v)
                            else:
                                episode_data[k].append(v)
            step4_time = time.time() - step4_start
            
            # Step 5: 转换数据
            step5_start = time.time()
            converted_data = self._convert_episode_data(episode_data, task['episode_idx'], config)
            step5_time = time.time() - step5_start
            
            # 清理内存
            del episode_data, batches, dataset, subset, dataloader
            gc.collect()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            timing_info = {
                'total': end_time - start_time,
                'load_dataset': step1_time,
                'create_subset': step2_time,
                'collect_batches': step3_time,
                'merge_batches': step4_time,
                'convert': step5_time,
                'memory_peak_mb': peak / 1024 / 1024,
                'memory_current_mb': end_memory,
                'memory_start_mb': start_memory,
            }
            
            self.logger.info(f"  Episode {task['episode_idx']}: "
                           f"total={timing_info['total']:.1f}s, "
                           f"load={timing_info['load_dataset']:.1f}s, "
                           f"collect={timing_info['collect_batches']:.1f}s, "
                           f"peak_mem={timing_info['memory_peak_mb']:.0f}MB")
            
            return {
                'episode_idx': task['episode_idx'],
                'data': converted_data,
                'success': True,
                'duration': end_time - start_time,
                'timing': timing_info,
            }
            
        except Exception as e:
            return {'episode_idx': task.get('episode_idx', -1), 'error': str(e), 'success': False}
    
    def _convert_episode_data(self, episode_data: Dict[str, Any], episode_idx: int, config: 'ConversionConfig'):
        """转换 episode 数据"""
        batch = episode_data.copy()
        
        # 添加 episode_index
        if 'episode_index' in batch:
            batch['episode_index'] = torch.full_like(batch['episode_index'], episode_idx)
        
        # 处理任务索引
        if 'task' in batch and 'task_index' not in batch:
            if 'task' in batch and len(batch['task']) > 0:
                batch['task_index'] = torch.zeros(len(batch['task']), dtype=torch.int32)
            if 'task' in batch:
                del batch['task']
        
        # 转换数据格式
        converted_batch = {}
        features = getattr(episode_data.get('_metadata', {}), 'features', {}) if hasattr(episode_data.get('_metadata', {}), 'features') else {}
        
        for key, value in batch.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.array(value)
            
            converted_batch[key] = value
        
        return converted_batch
    
    @abstractmethod
    def _get_num_episodes(self, dataset: Any) -> int:
        """获取数据集中的episode数量"""
        pass
    
    @abstractmethod
    def _get_episode_data(self, dataset: Any, episode_idx: int) -> Any:
        """获取单个episode的原始数据"""
        pass
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            self.logger.warning(f"无法序列化对象类型 {type(obj)}，转换为字符串")
            return str(obj)
