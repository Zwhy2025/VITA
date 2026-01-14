"""
配置管理工具
"""
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml
import json
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，支持 YAML 和 JSON 格式
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        logger.info(f"保存配置文件: {config_path}")
        
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise


def create_conversion_config(
    format: str,
    output_root: Optional[Union[str, Path]] = None,
    episodes: Optional[list] = None,
    remove_keys: Optional[list] = None,
    image_size: Optional[tuple] = None,
    batch_size: int = 16,
    num_workers: int = 8,
    compressors: str = 'disk'
) -> Dict[str, Any]:
    """创建转换配置
    
    Args:
        format: 数据格式 ('lerobot', 'mcap')
        output_root: 输出根目录
        episodes: 要转换的 episodes 列表
        remove_keys: 要移除的键列表
        image_size: 图像尺寸 (height, width)
        batch_size: 批处理大小
        num_workers: 工作进程数
        compressors: 压缩方式
        
    Returns:
        转换配置字典
    """
    config = {
        'format': format,
        'output_root': str(output_root) if output_root else None,
        'episodes': episodes,
        'remove_keys': remove_keys or [],
        'image_size': image_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'compressors': compressors,
    }
    
    # 移除 None 值
    config = {k: v for k, v in config.items() if v is not None}
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_fields = ['format']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"缺少必需配置项: {field}")
            return False
    
    # 验证格式
    valid_formats = ['lerobot', 'mcap']
    if config['format'] not in valid_formats:
        logger.error(f"不支持的格式: {config['format']}. 支持的格式: {valid_formats}")
        return False
    
    # 验证可选字段
    if 'episodes' in config and config['episodes'] is not None:
        if not isinstance(config['episodes'], list):
            logger.error("episodes 必须是列表")
            return False
    
    if 'image_size' in config and config['image_size'] is not None:
        if not isinstance(config['image_size'], (list, tuple)) or len(config['image_size']) != 2:
            logger.error("image_size 必须是包含2个元素的列表或元组")
            return False
    
    logger.info("配置验证通过")
    return True