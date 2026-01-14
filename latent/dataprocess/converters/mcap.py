"""
MCAP 数据格式转换器
"""
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import numpy as np
import cv2
from collections import defaultdict
import logging

try:
    import mcap
    from mcap.reader import make_reader
    from mcap.schema import Schema
    from mcap.records import Message
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False
    mcap = None

from .base import BaseDataConverter, ConversionConfig


class MCAPConverter(BaseDataConverter):
    """MCAP 格式数据转换器"""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        super().__init__(config)
        if not MCAP_AVAILABLE:
            raise ImportError("需要安装 mcap: pip install mcap")
    
    def detect_format(self, path: Union[str, Path]) -> bool:
        """检测是否为 MCAP 格式"""
        path = Path(path)
        
        # 检查是否为 .mcap 文件或包含 .mcap 文件的目录
        if path.is_file() and path.suffix == '.mcap':
            return True
        
        if path.is_dir():
            mcap_files = list(path.glob('*.mcap'))
            return len(mcap_files) > 0
        
        return False
    
    def load_datasets(self, path: Union[str, Path]) -> List[Any]:
        """加载 MCAP 数据集"""
        path = Path(path)
        
        if path.is_file():
            mcap_files = [path]
        else:
            mcap_files = list(path.glob('*.mcap'))
        
        if not mcap_files:
            raise ValueError(f"在 {path} 中未找到 MCAP 文件")
        
        datasets = []
        for mcap_file in mcap_files:
            dataset = MCAPDataset(mcap_file, self.config)
            datasets.append(dataset)
            self.logger.info(f"加载 MCAP 文件: {mcap_file}")
        
        return datasets
    
    def extract_metadata(self, datasets: List[Any], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """提取 MCAP 数据集元数据"""
        if not datasets:
            raise ValueError("数据集列表为空")
        
        path = Path(path) if path else None
        
        # 合并所有数据集的信息
        total_frames = 0
        total_episodes = 0
        all_features = {}
        fps = 30  # 默认帧率
        
        for dataset in datasets:
            metadata = dataset.get_metadata()
            total_frames += metadata.get('num_frames', 0)
            total_episodes += metadata.get('num_episodes', 0)
            all_features.update(metadata.get('features', {}))
            fps = metadata.get('fps', fps)
        
        # 移除指定的键
        features = {k: v for k, v in all_features.items() if k not in self.config.remove_keys}
        
        # 识别图像键
        image_keys = [k for k, v in features.items() if v.get('dtype') == 'image']
        
        return {
            "repo_id": f"mcap_dataset_{len(datasets)}_files",
            "stats": {},  # MCAP 数据集通常需要单独计算统计信息
            "num_frames": total_frames,
            "num_episodes": total_episodes,
            "features": features,
            "camera_keys": image_keys,
            "video_keys": [],
            "image_keys": image_keys,
            "fps": fps,
            "tasks": {},
        }
    
    def convert_episode(self, episode_data: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """转换单个 episode 数据"""
        converted_data = {}
        
        for key, value in episode_data.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, np.ndarray):
                if value.dtype == np.uint8 and len(value.shape) == 3:
                    # 图像数据，可能需要调整尺寸
                    if self.config.image_size is not None:
                        value = cv2.resize(value, self.config.image_size[::-1])
                converted_data[key] = value
            else:
                converted_data[key] = np.array(value)
        
        # 添加 episode_index
        converted_data['episode_index'] = np.full(len(next(iter(converted_data.values()))), episode_idx)
        
        return converted_data
    
    def _get_num_episodes(self, dataset: Any) -> int:
        """获取数据集中的 episode 数量"""
        return dataset.get_num_episodes()
    
    def _get_episode_data(self, dataset: Any, episode_idx: int) -> Dict[str, Any]:
        """获取单个 episode 的数据"""
        return dataset.get_episode_data(episode_idx)


class MCAPDataset:
    """MCAP 数据集包装器"""
    
    def __init__(self, mcap_file: Path, config: ConversionConfig):
        self.mcap_file = mcap_file
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{mcap_file.name}")
        
        # 初始化读取器
        self.reader = make_reader(str(mcap_file))
        self._metadata = None
        self._episodes = None
        
        # 解析数据
        self._parse_mcap()
    
    def _parse_mcap(self):
        """解析 MCAP 文件"""
        self.logger.info("解析 MCAP 文件...")
        
        # 获取所有消息
        messages = list(self.reader.iter_messages())
        
        # 按时间戳和主题组织数据
        self.data_by_topic = defaultdict(list)
        self.timestamps = []
        
        for msg in messages:
            topic = msg.topic
            data = msg.data
            timestamp = msg.log_time
            
            self.data_by_topic[topic].append((timestamp, data))
            self.timestamps.append(timestamp)
        
        # 对每个主题的数据按时间排序
        for topic in self.data_by_topic:
            self.data_by_topic[topic].sort(key=lambda x: x[0])
        
        # 获取唯一时间戳并排序
        self.timestamps = sorted(set(self.timestamps))
        
        self.logger.info(f"解析完成: {len(self.data_by_topic)} 个主题, {len(self.timestamps)} 个时间戳")
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取数据集元数据"""
        if self._metadata is None:
            self._metadata = self._compute_metadata()
        return self._metadata
    
    def _compute_metadata(self) -> Dict[str, Any]:
        """计算元数据"""
        features = {}
        
        for topic, data_list in self.data_by_topic.items():
            if not data_list:
                continue
            
            # 获取第一个数据样本
            _, sample_data = data_list[0]
            
            if isinstance(sample_data, np.ndarray):
                if len(sample_data.shape) == 3 and sample_data.shape[-1] == 3:
                    # 图像数据
                    features[topic] = {
                        'dtype': 'image',
                        'shape': list(sample_data.shape),
                    }
                else:
                    # 其他数组数据
                    features[topic] = {
                        'dtype': 'array',
                        'shape': list(sample_data.shape),
                    }
            else:
                # 标量数据
                features[topic] = {
                    'dtype': 'scalar',
                    'shape': [1],
                }
        
        # 估算 episodes 数量（这里简化处理，实际可能需要更复杂的逻辑）
        num_episodes = 1  # MCAP 文件通常包含连续记录
        num_frames = len(self.timestamps)
        
        return {
            'num_episodes': num_episodes,
            'num_frames': num_frames,
            'features': features,
            'fps': 30,  # 默认帧率
        }
    
    def get_num_episodes(self) -> int:
        """获取 episode 数量"""
        if self._episodes is None:
            metadata = self.get_metadata()
            self._episodes = metadata.get('num_episodes', 1)
        return self._episodes
    
    def get_episode_data(self, episode_idx: int) -> Dict[str, Any]:
        """获取 episode 数据"""
        episode_data = {}
        
        for topic, data_list in self.data_by_topic.items():
            # 提取该主题的所有数据
            topic_data = []
            for _, data in data_list:
                if isinstance(data, np.ndarray):
                    topic_data.append(data)
                else:
                    topic_data.append(np.array(data))
            
            if topic_data:
                # 堆叠数据
                try:
                    episode_data[topic] = np.stack(topic_data)
                except ValueError:
                    # 如果无法堆叠，使用第一个样本的形状
                    episode_data[topic] = np.array(topic_data)
        
        # 添加元数据
        episode_data['_metadata'] = {
            'features': self.get_metadata()['features'],
            'tasks': {},
        }
        
        return episode_data