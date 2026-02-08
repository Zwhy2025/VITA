import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import zarr
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_EPISODE_IDX = 30
ENABLE_CSV_EXPORT = True
ENABLE_JOINT_PLOT = True
JOINTS_TO_PLOT = [7, 10]
OUTPUT_DIR_NAME = "outputs/visualize"

def load_config(dataset_path: Path) -> dict:
    config_path = dataset_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def get_episode_range(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(f"Episode index {episode_idx} out of range [0, {len(episode_ends)-1}]")
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    return int(start_idx), int(end_idx)

def organize_camera_views(images_dict):
    left_keys = sorted([k for k in images_dict.keys() if "left" in k.lower()])
    right_keys = sorted([k for k in images_dict.keys() if "right" in k.lower()])
    center_keys = sorted([k for k in images_dict.keys() if k not in left_keys and k not in right_keys])
    return [images_dict[k] for k in left_keys + center_keys + right_keys]

def export_to_csv(data_group, start_idx, end_idx, output_csv_path, features_config):
    csv_data = {}
    keys_to_export = ["timestamp", "index", "frame_index", "episode_index", "action", "observation.state"]
    
    available_keys = [k for k in keys_to_export if k in data_group]
    if features_config:
        available_keys.extend([k for k in features_config.keys() 
                              if k not in available_keys and "image" not in k and "video" not in k and k in data_group])
    
    for key in available_keys:
        data = data_group[key][start_idx:end_idx]
        if data.ndim > 1:
            # 如果是 (N, 1) 形状，展平为一维并使用原始键名
            if data.shape[1] == 1:
                csv_data[key] = data[:, 0]
            else:
                # 多维数据，为每个维度创建列
                for dim in range(data.shape[1]):
                    col_name = f"{key}_{dim}"
                    if features_config and key in features_config and "names" in features_config[key] and features_config[key]["names"]:
                        try:
                            col_name = features_config[key]["names"][dim]
                        except IndexError:
                            pass
                    csv_data[col_name] = data[:, dim]
        else:
            csv_data[key] = data

    if not csv_data:
        return None

    df = pd.DataFrame(csv_data)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df

def generate_quality_report(df):
    if df is None or df.empty:
        return None
    
    joint_cols = [c for c in df.columns if any(x in c.lower() for x in ['state', 'joint', 'action'])]
    joint_cols = [c for c in joint_cols if '_s_' in c.lower()]
    
    if not joint_cols:
        return None
    
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, np.finfo(float).eps, dt)
    else:
        logger.warning("No 'timestamp' column found, using dt=1 for velocity calculation")
        dt = np.ones(len(df) - 1)
    
    report_data = []
    
    for col in joint_cols:
        values = df[col].values
        diffs = np.diff(values)
        velocities = np.abs(diffs / dt)
        
        velocity_changes = np.diff(diffs / dt)
        dt_for_accel = dt[1:]
        dt_for_accel = np.where(dt_for_accel == 0, np.finfo(float).eps, dt_for_accel)
        accelerations = velocity_changes / dt_for_accel
        accel_abs = np.abs(accelerations)
        
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        sign_change_ratio = sign_changes / len(diffs) if len(diffs) > 0 else 0
        
        velocity_threshold = np.mean(velocities) * 2
        velocity_jumps = np.sum(np.abs(np.diff(velocities)) > velocity_threshold)
        velocity_jump_ratio = velocity_jumps / len(velocities) if len(velocities) > 0 else 0
        
        report_data.append({
            'joint_name': col,
            'diff_mean': float(np.mean(np.abs(diffs))),
            'diff_max': float(np.max(np.abs(diffs))),
            'diff_min': float(np.min(np.abs(diffs))),
            'diff_median': float(np.median(np.abs(diffs))),
            'velocity_mean': float(np.mean(velocities)),
            'velocity_max': float(np.max(velocities)),
            'velocity_std': float(np.std(velocities)),
            'accel_mean': float(np.mean(accel_abs)),
            'accel_max': float(np.max(accel_abs)),
            'sign_change_ratio': float(sign_change_ratio),
            'velocity_jump_ratio': float(velocity_jump_ratio),
        })
    
    report_df = pd.DataFrame(report_data)
    return report_df

def plot_quality_report(report_df, output_plot_path):
    if report_df is None or report_df.empty:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Joint Quality Report', fontsize=16, fontweight='bold')
    
    joint_names = report_df['joint_name'].values
    x_pos = np.arange(len(joint_names))
    width = 0.2
    
    ax1 = axes[0]
    ax1.bar(x_pos - 1.5*width, report_df['diff_mean'], width, label='Mean', alpha=0.8)
    ax1.bar(x_pos - 0.5*width, report_df['diff_max'], width, label='Max', alpha=0.8)
    ax1.bar(x_pos + 0.5*width, report_df['diff_median'], width, label='Median', alpha=0.8)
    ax1.bar(x_pos + 1.5*width, report_df['diff_min'], width, label='Min', alpha=0.8)
    ax1.set_xlabel('Joint', fontsize=12)
    ax1.set_ylabel('Difference Value', fontsize=12)
    ax1.set_title('Difference Statistics (Mean/Max/Median/Min)', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(joint_names, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.bar(x_pos - width/2, report_df['velocity_mean'], width, label='Mean Velocity', alpha=0.8)
    ax2.bar(x_pos + width/2, report_df['velocity_max'], width, label='Max Velocity', alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, report_df['velocity_std'], 'ro-', label='Velocity Std', markersize=6)
    ax2.set_xlabel('Joint', fontsize=12)
    ax2.set_ylabel('Velocity Value', fontsize=12)
    ax2_twin.set_ylabel('Velocity Std', fontsize=12, color='r')
    ax2.set_title('Velocity Statistics (Mean/Max) and Std (Continuity)', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(joint_names, rotation=45, ha='right')
    ax2.legend(loc='upper left', fontsize=11)
    ax2_twin.legend(loc='upper right', fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_joints(df, output_plot_path):
    if df is None or df.empty:
        return
    
    joint_cols = [c for c in df.columns if any(x in c.lower() for x in ['state', 'joint', 'action'])]
    joint_cols = [c for c in joint_cols if '_s_' in c.lower()]
    
    if not joint_cols:
        return

    if JOINTS_TO_PLOT is None:
        return
    
    if isinstance(JOINTS_TO_PLOT, list):
        selected_cols = []
        for idx in JOINTS_TO_PLOT:
            if isinstance(idx, int) and 0 <= idx < len(joint_cols):
                selected_cols.append(joint_cols[idx])
        
        if not selected_cols:
            return
    else:
        return
    
    plt.figure(figsize=(12, 6))
    for col in selected_cols:
        plt.plot(df.index, df[col], marker='o', markersize=3, label=col)
    plt.title(f"Joint Values ({len(selected_cols)} joints)")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot_path)
    plt.close()

def _resolve_output_path(
    output_dir: Path,
    output_arg: str | None,
    default_name: str | None,
) -> Path | None:
    if output_arg is None:
        if default_name is None:
            return None
        return output_dir / default_name
    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = output_dir / output_path
    return output_path

def visualize_dataset(
    dataset_path: str,
    episode_idx: int = DEFAULT_EPISODE_IDX,
    output_dir: str | Path | None = None,
    video_output: str | None = None,
    csv_output: str | None = None,
):
    path = Path(dataset_path).resolve()
    if not path.exists():
        logger.error(f"Dataset path not found: {path}")
        return

    dataset_name = path.name
    if output_dir is None:
        output_dir = Path.cwd() / OUTPUT_DIR_NAME / dataset_name
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
    try:
        root = zarr.open(str(path), mode="r")
    except Exception as e:
        logger.error(f"Failed to open Zarr dataset: {e}")
        return

    if "data" not in root or "meta" not in root:
        logger.error("Invalid Zarr structure.")
        return

    data_group = root["data"]
    meta_group = root["meta"]
    
    if "episode_ends" not in meta_group:
        logger.error("'episode_ends' not found.")
        return
    
    episode_ends = meta_group["episode_ends"][:]
    if episode_idx >= len(episode_ends):
        logger.error(f"Episode index {episode_idx} exceeds total episodes {len(episode_ends)}.")
        return

    start_idx, end_idx = get_episode_range(episode_ends, episode_idx)
    num_frames = end_idx - start_idx

    video_output_path = _resolve_output_path(
        output_dir,
        video_output,
        f"ep{episode_idx}_video.mp4",
    )
    if video_output_path is None:
        logger.error("Video output path resolved to None.")
        return
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    features_config = config.get("features", {})
    
    enable_csv = ENABLE_CSV_EXPORT or csv_output is not None
    csv_output_path = _resolve_output_path(
        output_dir,
        csv_output if enable_csv else None,
        f"ep{episode_idx}_data.csv" if enable_csv else None,
    )
    if csv_output_path is not None:
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    df = export_to_csv(
        data_group,
        start_idx,
        end_idx,
        str(csv_output_path) if csv_output_path is not None else None,
        features_config,
    )
    
    if df is not None:
        report_df = generate_quality_report(df)
        
        if report_df is not None:
            quality_plot_path = output_dir / f"ep{episode_idx}_quality_plot.png"
            plot_quality_report(report_df, str(quality_plot_path))
        
        if ENABLE_JOINT_PLOT:
            joints_plot_path = output_dir / f"ep{episode_idx}_joints.png"
            plot_joints(df, str(joints_plot_path))

    camera_keys = config.get("camera_keys", [])
    if not camera_keys:
        camera_keys = [key for key in data_group.keys() if "image" in key and data_group[key].ndim == 4]
    
    if not camera_keys:
        logger.error("No camera keys found.")
        return
    
    fps = config.get("fps", 30)
    fps = int(fps) if isinstance(fps, (int, float)) else 30
    logger.info(f"Using FPS from dataset config: {fps} (video will be rendered at {fps} Hz)")

    writer = None
    for i in tqdm(range(num_frames), desc="Rendering Video"):
        global_idx = start_idx + i
        images_dict = {}
        
        for key in camera_keys:
            if key not in data_group:
                continue
                
            img_frame = data_group[key][global_idx]
            if img_frame.shape[0] == 3:
                img_frame = np.transpose(img_frame, (1, 2, 0))
            
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
            label = key.split(".")[-1]
            cv2.putText(img_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            images_dict[label] = img_frame
        
        if not images_dict:
            continue

        ordered_images = organize_camera_views(images_dict)
        max_h = max(img.shape[0] for img in ordered_images)
        resized_images = []
        for img in ordered_images:
            if img.shape[0] != max_h:
                w = int(img.shape[1] * max_h / img.shape[0])
                img = cv2.resize(img, (w, max_h))
            resized_images.append(img)
            
        concat_img = np.hstack(resized_images)
        info_text = f"Ep: {episode_idx} | Frame: {i}/{num_frames} | Global: {global_idx}"
        cv2.putText(concat_img, info_text, (10, max_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if writer is None:
            h, w = concat_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (w, h))
        
        writer.write(concat_img)

    if writer:
        writer.release()
    else:
        logger.error("No frames were written.")

def main():
    parser = argparse.ArgumentParser(description="Visualize Zarr Dataset")
    parser.add_argument("dataset_path", type=str, help="Path to Zarr dataset directory")
    parser.add_argument(
        "--episode-idx",
        "--episode_idx",
        dest="episode_idx",
        type=int,
        default=DEFAULT_EPISODE_IDX,
        help="Episode index to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Base output directory. Defaults to outputs/visualize/<dataset>.",
    )
    parser.add_argument(
        "--output",
        dest="video_output",
        type=str,
        default=None,
        help="Video output path. Relative paths are placed in the output dir.",
    )
    parser.add_argument(
        "--csv-output",
        "--csv_output",
        dest="csv_output",
        type=str,
        default=None,
        help="CSV output path. Relative paths are placed in the output dir.",
    )
    args = parser.parse_args()
    visualize_dataset(
        args.dataset_path,
        episode_idx=args.episode_idx,
        output_dir=args.output_dir,
        video_output=args.video_output,
        csv_output=args.csv_output,
    )

if __name__ == "__main__":
    main()
