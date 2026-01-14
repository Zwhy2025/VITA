"""
数据转换器工厂模式
"""
from pathlib import Path
from typing import Union, Type, Optional
import logging

from .base import BaseDataConverter, ConversionConfig
from .lerobot import LeRobotConverter, OptimizedLeRobotConverter, TorchCodecGPUConverter
from .mcap import MCAPConverter


class ConverterFactory:
    """转换器工厂类"""
    
    _converters: dict[str, Type[BaseDataConverter]] = {}
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def register_converter(cls, format_name: str, converter_class: Type[BaseDataConverter]):
        """注册转换器"""
        cls._converters[format_name.lower()] = converter_class
        cls._logger.info(f"注册转换器: {format_name} -> {converter_class.__name__}")
    
    @classmethod
    def create_converter(cls, format_name: str, config: Optional[ConversionConfig] = None) -> BaseDataConverter:
        """创建转换器"""
        format_name = format_name.lower()
        
        if format_name not in cls._converters:
            raise ValueError(f"未知的数据格式: {format_name}. 支持的格式: {list(cls._converters.keys())}")
        
        converter_class = cls._converters[format_name]
        return converter_class(config)
    
    @classmethod
    def detect_and_create(cls, path: Union[str, Path], config: Optional[ConversionConfig] = None) -> BaseDataConverter:
        """自动检测格式并创建转换器"""
        path = Path(path)
        
        # 按优先级顺序检测格式
        for format_name, converter_class in cls._converters.items():
            try:
                converter = converter_class(config)
                if converter.detect_format(path):
                    cls._logger.info(f"检测到格式: {format_name} (路径: {path})")
                    return converter
            except Exception as e:
                cls._logger.debug(f"格式检测失败 {format_name}: {e}")
                continue
        
        raise ValueError(f"无法检测数据格式: {path}. 支持的格式: {list(cls._converters.keys())}")
    
    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """获取支持的格式列表"""
        return list(cls._converters.keys())


# 注册内置转换器
def _register_builtin_converters():
    """注册内置转换器"""
    ConverterFactory.register_converter('lerobot', LeRobotConverter)
    ConverterFactory.register_converter('mcap', MCAPConverter)
    ConverterFactory.register_converter('lerobot_fast', OptimizedLeRobotConverter)
    ConverterFactory.register_converter('lerobot_gpu', TorchCodecGPUConverter)


# 自动注册
_register_builtin_converters()