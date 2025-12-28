import subprocess
import time
import re
from collections import deque
import logging
import argparse
import psutil  # 需安装：pip install psutil（监控CPU/内存）

# 配置日志（关闭自动换行，保证格式统一）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # 固定时间格式，减少长度波动
)

def get_gpu_count():
    """自动检测当前机器的GPU总数"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpu_indices = [line.strip() for line in result.strip().split('\n') if line.strip()]
        return len(gpu_indices)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("错误：无法检测GPU数量，请确保已安装NVIDIA驱动和nvidia-smi")
        return 0

def clean_metric_value(value):
    """
    清洗指标值：移除括号、空格，处理N/A/[N/A]等无效值
    :param value: 原始指标值（字符串）
    :return: 清洗后的值，无效值返回None
    """
    # 移除所有括号、空格，转小写
    cleaned = re.sub(r'[\[\] ]', '', value).lower()
    # 处理无效值
    if cleaned in ['na', 'n/a', '']:
        return None
    return cleaned

def get_gpu_full_metrics(gpu_id=0):
    """
    获取指定GPU的核心指标（仅保留利用率、显存，移除温度、功耗）
    :return: 字典，包含所有GPU指标
    """
    try:
        # 仅查询需要的指标：利用率、显存使用、显存总量（移除温度、功耗等）
        result = subprocess.check_output(
            [
                "nvidia-smi", "-i", str(gpu_id),
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            encoding="utf-8"
        )
        # 解析结果（逗号分隔，注意处理空格）
        raw_metrics = [item.strip() for item in result.strip().split(',')]
        if len(raw_metrics) != 3:
            logging.warning(f"GPU {gpu_id} 返回的指标数量异常：{len(raw_metrics)}（预期3）")
            return None
        
        # 逐个清洗指标值并转换类型
        metrics = []
        for idx, val in enumerate(raw_metrics):
            cleaned_val = clean_metric_value(val)
            if cleaned_val is None:
                # 不同指标的默认值
                if idx == 0:  # 利用率（整数）
                    metrics.append(0)
                else:  # 显存（浮点数）
                    metrics.append(0.0)
            else:
                # 转换类型：整数/浮点数
                if idx == 0:  # 利用率
                    metrics.append(int(cleaned_val))
                else:  # 显存
                    metrics.append(float(cleaned_val))
        
        # 整理为字典，统一单位（显存转GB）
        return {
            "gpu_id": gpu_id,
            "utilization": metrics[0],  # %
            "mem_used": round(metrics[1] / 1024, 2),  # GB
            "mem_total": round(metrics[2] / 1024, 2),  # GB
            "mem_util": round((metrics[1] / metrics[2] * 100) if metrics[2] != 0 else 0, 2),  # 显存利用率%
        }
    except Exception as e:
        logging.error(f"获取GPU {gpu_id} 指标失败：{e}")
        return None

def get_system_metrics():
    """获取系统CPU/内存指标"""
    try:
        return {
            "cpu_util": round(psutil.cpu_percent(interval=0.01), 1),  # CPU整体利用率%
            "mem_used": round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 2),  # GB
            "mem_total": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),  # GB
            "mem_util": round(psutil.virtual_memory().percent, 1)  # 内存利用率%
        }
    except Exception as e:
        logging.error(f"获取系统指标失败：{e}")
        return {"cpu_util": 0, "mem_used": 0, "mem_total": 0, "mem_util": 0}

def monitor_all_metrics(total_seconds=60, interval=0.1):
    """
    自动检测所有GPU，同步监控GPU核心指标+系统指标（仅保留利用率、显存，优化格式化输出）
    """
    # 1. 检测GPU数量
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        logging.warning("未检测到可用GPU，监控终止")
        return {}
    
    # 2. 初始化每个GPU的指标队列（存储所有采样数据）
    sample_count = int(round(total_seconds / interval))
    gpu_metrics_history = {
        gpu_id: deque(maxlen=sample_count) 
        for gpu_id in range(gpu_count)
    }
    system_metrics_history = deque(maxlen=sample_count)
    
    logging.info(f"检测到 {gpu_count} 个GPU，开始全指标监控（持续{total_seconds}秒，采样间隔{interval}秒）...")
    start_time = time.time()
    
    try:
        for step in range(sample_count):
            # 3. 采集所有GPU指标
            current_gpu_metrics = {}
            for gpu_id in range(gpu_count):
                gpu_metrics = get_gpu_full_metrics(gpu_id)
                if gpu_metrics:
                    gpu_metrics_history[gpu_id].append(gpu_metrics)
                    current_gpu_metrics[gpu_id] = gpu_metrics
            
            # 4. 采集系统指标
            system_metrics = get_system_metrics()
            system_metrics_history.append(system_metrics)
            
            # 5. 打印实时监控信息（固定宽度格式化，避免抖动）
            elapsed = time.time() - start_time
            # 拼接GPU实时信息（固定字段宽度，右对齐）
            gpu_info_list = []
            for gpu_id, metrics in current_gpu_metrics.items():
                # 每个字段固定宽度，不足补空格，超过截断（保证长度一致）
                gpu_info = (
                    f"GPU{gpu_id}: 利用率{metrics['utilization']:3d}% | "  # 3位宽度（0-100）
                    f"显存{metrics['mem_used']:6.2f}/{metrics['mem_total']:5.1f}GB({metrics['mem_util']:5.2f}%)"  # 显存固定宽度
                )
                gpu_info_list.append(gpu_info)
            
            # 拼接系统实时信息（固定宽度）
            system_info = (
                f"系统: CPU{system_metrics['cpu_util']:4.1f}% | "  # 4位宽度（0.0-100.0）
                f"内存{system_metrics['mem_used']:7.2f}/{system_metrics['mem_total']:6.2f}GB({system_metrics['mem_util']:4.1f}%)"  # 内存固定宽度
            )
            
            # 合并所有信息，空GPU时补占位符
            gpu_info_str = ' || '.join(gpu_info_list) if gpu_info_list else ' ' * 60  # 空时补固定长度空格
            log_msg = f"已运行 {elapsed:6.2f}秒 | {gpu_info_str} || {system_info}"
            
            # 打印
            logging.info(log_msg)
            
            # 6. 精准控制采样间隔
            expected_next = (step + 1) * interval
            sleep_time = max(0, expected_next - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logging.info("\n监控被手动终止")
    
    # 7. 输出最终统计结果（格式化，仅保留利用率、显存）
    logging.info("\n===== 监控汇总结果 =====")
    for gpu_id in range(gpu_count):
        history = gpu_metrics_history[gpu_id]
        if not history:
            logging.info(f"GPU {gpu_id}：无有效监控数据")
            continue
        
        # 计算关键指标的统计值
        util_list = [m['utilization'] for m in history]
        mem_util_list = [m['mem_util'] for m in history]
        
        # 格式化统计输出（固定宽度，对齐显示）
        logging.info(f"\nGPU {gpu_id} 统计：")
        logging.info(f"  - GPU利用率    ：平均 {sum(util_list)/len(util_list):6.2f}% | 最高 {max(util_list):3d}% | 最低 {min(util_list):3d}%")
        logging.info(f"  - 显存利用率  ：平均 {sum(mem_util_list)/len(mem_util_list):6.2f}% | 最高 {max(mem_util_list):6.2f}% | 最低 {min(mem_util_list):6.2f}%")
    
    # 系统指标统计（格式化）
    if system_metrics_history:
        cpu_list = [m['cpu_util'] for m in system_metrics_history]
        mem_util_list = [m['mem_util'] for m in system_metrics_history]
        logging.info(f"\n系统统计：")
        logging.info(f"  - CPU利用率    ：平均 {sum(cpu_list)/len(cpu_list):6.2f}% | 最高 {max(cpu_list):4.1f}% | 最低 {min(cpu_list):4.1f}%")
        logging.info(f"  - 内存利用率  ：平均 {sum(mem_util_list)/len(mem_util_list):6.2f}% | 最高 {max(mem_util_list):4.1f}% | 最低 {min(mem_util_list):4.1f}%")
    
    return {"gpu_metrics": gpu_metrics_history, "system_metrics": system_metrics_history}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动检测并监控所有GPU+系统核心指标（仅保留利用率、显存）")
    parser.add_argument(
        "-t", "--total-seconds",
        type=float,
        default=300,
        help="总监控时长（秒），默认30"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=0.1,  # 采样间隔稍放宽，避免nvidia-smi调用过频
        help="采样间隔（秒），默认0.1"
    )
    args = parser.parse_args()
    
    # 安装依赖提示
    try:
        import psutil
    except ImportError:
        logging.error("请先安装psutil：pip install psutil")
        exit(1)
    
    monitor_all_metrics(
        total_seconds=args.total_seconds,
        interval=args.interval
    )