"""
数据验证工具
"""
from pathlib import Path
from typing import Dict, List, Any, Union
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


def validate_dataset(dataset_path: Union[str, Path]) -> bool:
    """验证数据集格式和完整性
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        是否验证通过
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    logger.info(f"开始验证数据集: {dataset_path}")
    
    try:
        # 检测格式并验证
        from ..converters import ConverterFactory
        
        converter = ConverterFactory.detect_and_create(dataset_path)
        datasets = converter.load_datasets(dataset_path)
        
        # 验证每个数据集
        for i, dataset in enumerate(datasets):
            if not _validate_single_dataset(dataset, f"数据集 {i+1}"):
                return False
        
        logger.info("数据集验证通过")
        return True
        
    except Exception as e:
        logger.error(f"数据集验证失败: {e}")
        return False


def validate_conversion(output_path: Union[str, Path]) -> bool:
    """验证转换后的数据
    
    Args:
        output_path: 输出路径
        
    Returns:
        是否验证通过
    """
    output_path = Path(output_path)
    
    if not output_path.exists():
        logger.error(f"输出路径不存在: {output_path}")
        return False
    
    logger.info(f"开始验证转换结果: {output_path}")
    
    try:
        # 检查必要文件
        config_file = output_path / "config.json"
        if not config_file.exists():
            logger.error("缺少 config.json 文件")
            return False
        
        # 检查 zarr 结构
        try:
            import zarr
            root = zarr.open(str(output_path))
            
            if 'data' not in root:
                logger.error("缺少 data 组")
                return False
            
            if 'meta' not in root:
                logger.error("缺少 meta 组")
                return False
            
            if 'episode_ends' not in root['meta']:
                logger.error("缺少 episode_ends")
                return False
            
            logger.info("Zarr 结构验证通过")
            
        except Exception as e:
            logger.error(f"Zarr 结构验证失败: {e}")
            return False
        
        # 读取配置
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 验证数据一致性
        num_episodes = config.get('num_episodes', 0)
        features = config.get('features', {})
        
        if num_episodes == 0:
            logger.warning("没有 episodes")
        
        if not features:
            logger.warning("没有特征数据")
        
        # 验证数据文件
        if not _validate_zarr_data(output_path, config):
            return False
        
        logger.info("转换结果验证通过")
        return True
        
    except Exception as e:
        logger.error(f"转换结果验证失败: {e}")
        return False


def _validate_single_dataset(dataset: Any, dataset_name: str) -> bool:
    """验证单个数据集"""
    try:
        # 检查基本属性
        if not hasattr(dataset, 'num_episodes'):
            logger.error(f"{dataset_name}: 缺少 num_episodes 属性")
            return False
        
        if not hasattr(dataset, 'num_frames'):
            logger.error(f"{dataset_name}: 缺少 num_frames 属性")
            return False
        
        if dataset.num_episodes <= 0:
            logger.error(f"{dataset_name}: episodes 数量无效: {dataset.num_episodes}")
            return False
        
        if dataset.num_frames <= 0:
            logger.error(f"{dataset_name}: frames 数量无效: {dataset.num_frames}")
            return False
        
        # 检查数据访问
        try:
            if hasattr(dataset, 'features'):
                features = dataset.features
                if not features:
                    logger.warning(f"{dataset_name}: 没有特征数据")
        except Exception as e:
            logger.error(f"{dataset_name}: 无法访问特征数据: {e}")
            return False
        
        logger.info(f"{dataset_name}: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        return True
        
    except Exception as e:
        logger.error(f"{dataset_name}: 验证失败: {e}")
        return False


def _validate_zarr_data(output_path: Path, config: Dict[str, Any]) -> bool:
    """验证 zarr 数据"""
    try:
        import zarr
        root = zarr.open(str(output_path))
        
        data_group = root['data']
        meta_group = root['meta']
        episode_ends = meta_group['episode_ends'][:]
        
        num_episodes = len(episode_ends)
        expected_num_episodes = config.get('num_episodes', num_episodes)
        
        if num_episodes != expected_num_episodes:
            logger.warning(f"Episodes 数量不匹配: 实际 {num_episodes}, 期望 {expected_num_episodes}")
        
        # 验证每个特征的数据长度
        features = config.get('features', {})
        for feature_name, feature_info in features.items():
            if feature_name in data_group:
                data_array = data_group[feature_name]
                
                if len(data_array) != episode_ends[-1]:
                    logger.error(f"特征 {feature_name} 数据长度不匹配: {len(data_array)} vs {episode_ends[-1]}")
                    return False
                
                # 检查数据完整性
                if np.isnan(data_array[:]).any() and feature_info.get('dtype') != 'image':
                    logger.warning(f"特征 {feature_name} 包含 NaN 值")
        
        logger.info("Zarr 数据验证通过")
        return True
        
    except Exception as e:
        logger.error(f"Zarr 数据验证失败: {e}")
        return False