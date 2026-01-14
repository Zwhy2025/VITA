#!/usr/bin/env python3
"""
VITA 数据转换工具
支持 LeRobot 和 MCAP 格式的数据转换
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试添加 VITA 根目录到路径（用于查找 latent 模块）
vita_root = project_root.parent.parent
if str(vita_root) not in sys.path:
    sys.path.insert(0, str(vita_root))

try:
    from .converters import ConverterFactory, ConversionConfig
    from .utils import validate_dataset, validate_conversion, load_config, save_config
except ImportError:
    from converters import ConverterFactory, ConversionConfig
    from utils import validate_dataset, validate_conversion, load_config, save_config


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description="VITA 数据转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动检测格式并转换
  python -m latent.dataprocess.convert datasets/r2v2/dual_flip_box_video_50/
  
  # 指定格式转换
  python -m latent.dataprocess.convert datasets/r2v2/dual_flip_box_video_50/ --format lerobot
  
  # 使用配置文件
  python -m latent.dataprocess.convert --config example_config.yaml
  
  # 高级用法
  python -m latent.dataprocess.convert datasets/r2v2/dual_flip_box_video_50/ \\
    --episodes 0 1 2 \\
    --image-size 224 224 \\
    --batch-size 32 \\
    --validate-input \\
    --validate-output \\
    --verbose
        """
    )
    
    # 基本参数 - 使用 nargs='?' 使 input_path 可选（当使用 --list-formats 时）
    parser.add_argument(
        'input_path',
        nargs='?',
        default=None,
        help='输入数据路径（目录或文件）'
    )
    
    parser.add_argument(
        '--format',
        choices=['lerobot', 'mcap', 'auto'],
        default='auto',
        help='数据格式（默认：自动检测）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（默认：自动生成）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路径（YAML 或 JSON）'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        help='要转换的 episodes 列表（空表示全部）'
    )
    
    parser.add_argument(
        '--remove-keys',
        type=str,
        nargs='+',
        help='要移除的键列表'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        help='目标图像尺寸（高度 宽度）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='批处理大小（默认：16）'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='工作进程数（默认：1）'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=False,
        help='启用并行转换（注意：可能导致 OOM，建议使用 --num-workers=1）'
    )
    
    parser.add_argument(
        '--compressors',
        type=str,
        default='disk',
        choices=['default', 'disk'],
        help='压缩方式（默认：disk）'
    )
    
    parser.add_argument(
        '--validate-input',
        action='store_true',
        help='转换前验证输入数据'
    )
    
    parser.add_argument(
        '--validate-output',
        action='store_true',
        default=True,
        help='转换后验证输出数据（默认开启）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '--list-formats',
        action='store_true',
        help='列出支持的格式'
    )
    
    parser.add_argument(
        '--optimized', '-O',
        action='store_true',
        default=False,
        help='使用优化的转换器（PyAV + pandas，速度提升 50-60x）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # 列出支持的格式
        if args.list_formats:
            formats = ConverterFactory.get_supported_formats()
            print("支持的数据格式:")
            for fmt in formats:
                print(f"  - {fmt}")
            return 0
        
        # 加载配置文件
        config = None
        if args.config:
            file_config = load_config(args.config)
            config = ConversionConfig(
                output_root=file_config.get('output_root'),
                episodes=file_config.get('episodes'),
                remove_keys=file_config.get('remove_keys'),
                image_size=file_config.get('image_size'),
                batch_size=file_config.get('batch_size', 16),
                num_workers=file_config.get('num_workers', 8),
                compressors=file_config.get('compressors', 'disk'),
            )
        else:
            config = ConversionConfig(
                output_root=args.output,
                episodes=args.episodes,
                remove_keys=args.remove_keys,
                image_size=tuple(args.image_size) if args.image_size else None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                parallel=args.parallel,
                compressors=args.compressors,
            )
        
        # 检查输入路径
        if not args.input_path:
            print("错误：请提供输入路径")
            return 1
        
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"错误：输入路径不存在: {input_path}")
            return 1
        
        # 验证输入数据
        if args.validate_input:
            print("验证输入数据...")
            if not validate_dataset(input_path):
                print("❌ 输入数据验证失败")
                return 1
            print("✅ 输入数据验证通过")
        
        # 创建转换器
        if args.optimized:
            print("🚀 使用优化版转换器 (PyAV + pandas, 速度提升 50-60x)")
            converter = ConverterFactory.create_converter('lerobot_fast', config)
        elif args.format == 'auto':
            converter = ConverterFactory.detect_and_create(input_path, config)
        else:
            converter = ConverterFactory.create_converter(args.format, config)
        
        # 执行转换
        if args.parallel:
            print("⚠️  警告：并行模式可能导致 OOM（每个 episode 需要 ~11GB 内存）")
            print("💡 建议：不使用 --parallel 参数进行顺序转换，更稳定可靠")
        
        print("开始数据转换...")
        output_path = converter.convert(input_path, config.output_root)
        print(f"✅ 数据转换完成: {output_path}")
        
        # 验证输出数据
        if args.validate_output:
            print("验证输出数据...")
            if not validate_conversion(output_path):
                print("❌ 输出数据验证失败")
                return 1
            print("✅ 输出数据验证通过")
        
        print("🎉 所有操作完成！")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())