import argparse
from pathlib import Path
import os
import sys
import logging

current_file_path = Path(__file__).resolve()
root_path = current_file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from dataprocess.core.datasets import create_av_aloha_dataset_from_lerobot

REMOVE_KEYS = []
IMAGE_SIZE = None
TARGET_FPS = 20
VIDEO_DECODE_THREADS = os.cpu_count() -2 
VIDEO_DECODE_THREAD_TYPE = "FRAME"
COMPRESSOR = "default"

if 'FLARE_DATASETS_DIR' in os.environ:
    DEFAULT_OUTPUT_ROOT = Path(os.environ['FLARE_DATASETS_DIR'])
else:
    DEFAULT_OUTPUT_ROOT = Path(__file__).parent.parent.parent / "gym-av-aloha" / "outputs"

          
def convert_dataset(
    dataset_path: str,
    *,
    output_root: str | Path | None = None,
    compressor: str = COMPRESSOR,
    video_decode_threads: int | None = VIDEO_DECODE_THREADS,
    video_decode_thread_type: str | None = VIDEO_DECODE_THREAD_TYPE,
):
    remove_keys = REMOVE_KEYS
    image_size = IMAGE_SIZE
    target_fps = TARGET_FPS
    if video_decode_threads == 0:
        video_decode_threads = None

    path = Path(dataset_path).resolve()
    if not path.exists() or not path.is_dir():
        logger.error(f"Error: Dataset path not found: {path}")
        return

    if output_root is None:
        safe_id = path.name
        if target_fps is not None:
            safe_id = f"{safe_id}"
        output_root = DEFAULT_OUTPUT_ROOT / safe_id
    else:
        output_root = Path(output_root).resolve()

    logger.info(f"Converting Dataset: {path} -> {output_root}")

    create_av_aloha_dataset_from_lerobot(
        dataset_root=path,
        root=output_root,
        remove_keys=remove_keys,
        image_size=image_size,
        target_fps=target_fps,
        compressors=compressor,
        video_decode_threads=video_decode_threads,
        video_decode_thread_type=video_decode_thread_type,
    )

    logger.info(f"Conversion completed: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets from LeRobot format to AV-ALOHA format."
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Dataset local path (relative or absolute).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory.",
    )

    args = parser.parse_args()
    convert_dataset(
        args.dataset_path,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()
