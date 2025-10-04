import sys
import json
import logging
import warnings
from tqdm import tqdm
from typing import Generator, Iterable
from tagger.interrogator.interrogator import AbsInterrogator
from PIL import Image, ImageFile
from pathlib import Path
from typing import Dict
import argparse
import structlog
import cv2
import numpy as np

from tagger.interrogators import interrogators

# Allow images with broken headers to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Filter CUDA provider warning
warnings.filterwarnings('ignore', message='Specified provider .* is not in available provider names')

# Global progress bar reference for tqdm-safe logging
_current_progress_bar = None

class TqdmWriter:
    """Custom writer that uses tqdm.write() if progress bar is active"""
    def write(self, message):
        if _current_progress_bar is not None:
            tqdm.write(message.rstrip(), file=sys.stderr)
        else:
            sys.stderr.write(message)

    def flush(self):
        if _current_progress_bar is None:
            sys.stderr.flush()

# Configure structured logging with tqdm compatibility
def setup_logging(enable_progress_bar: bool = False):
    # Create custom handler that uses TqdmWriter
    handler = logging.StreamHandler(TqdmWriter())
    handler.setLevel(logging.INFO)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    return structlog.get_logger("wd14-tagger")

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dir', help='Predictions for all images in the directory')
group.add_argument('--file', help='Predictions for one file')

parser.add_argument(
    '--threshold',
    type=float,
    default=0.35,
    help='Prediction threshold (default is 0.35)')
parser.add_argument(
    '--ext',
    default='.txt',
    help='Extension to add to caption file in case of dir option (default is .txt)')
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Overwrite caption file if it exists')
parser.add_argument(
    '--cpu',
    action='store_true',
    help='Use CPU only')
parser.add_argument(
    '--rawtag',
    action='store_true',
    help='Use the raw output of the model')
parser.add_argument(
    '--recursive',
    action='store_true',
    help='Enable recursive file search')
parser.add_argument(
    '--exclude-tag',
    dest='exclude_tags',
    action='append',
    metavar='t1,t2,t3',
    help='Specify tags to exclude (Need comma-separated list)')
parser.add_argument(
    '--additional-tag',
    dest='additional_tags',
    action='append',
    metavar='t1,t2,t3',
    help='Specify additional tags (Need comma-separated list)')
parser.add_argument(
    '--model',
    default='wd14-convnextv2.v1',
    metavar='MODELNAME',
    help='modelname to use for prediction (default is wd14-convnextv2.v1)')
parser.add_argument(
    '--json',
    action='store_true',
    help='output json instead of plaintext'
)
parser.add_argument(
    '--progress-bar',
    action='store_true',
    help='Show progress bar instead of processing messages'
)
parser.add_argument(
    '--sample',
    default='30',
    help='Video sampling mode: "all" (every frame), "keyframes" (scene changes), or integer N (every Nth frame). Default: 30'
)
def parse_args():
    """Parse command line arguments."""
    return parser.parse_args()


def process_directory(args, logger, interrogator):
    """Process all images in a directory."""
    global _current_progress_bar

    root_path = Path(args.dir)
    # First collect all files to process for progress bar and logging
    media_files = list(explore_media_files(root_path, args.recursive))

    if not media_files:
        logger.warning("No media files found", directory=str(root_path))
        return

    logger.info("Starting batch processing",
                directory=str(root_path),
                total_files=len(media_files),
                model=args.model,
                threshold=args.threshold)

    if args.progress_bar:
        _current_progress_bar = tqdm(media_files, desc="Processing media")
        iterator = _current_progress_bar
    else:
        iterator = media_files

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for media_path in iterator:
        try:
            if is_video_file(media_path):
                # Process video file
                if not args.progress_bar:
                    logger.info("Processing video",
                               file=str(media_path.name),
                               path=str(media_path))

                frames_processed = process_video_file(
                    media_path, not args.rawtag, parse_exclude_tags(args),
                    parse_additional_tags(args), args, interrogator, logger)
                processed_count += frames_processed

            else:
                # Process image file
                caption_path = media_path.parent / f'{media_path.stem}{args.ext}'

                if caption_path.is_file() and not args.overwrite:
                    skipped_count += 1
                    if not args.progress_bar:
                        logger.info("Skipping existing file",
                                   file=str(media_path.name),
                                   caption_exists=True)
                    continue

                if not args.progress_bar:
                    logger.info("Processing image",
                               file=str(media_path.name),
                               path=str(media_path))

                tags = image_interrogate(media_path, not args.rawtag, parse_exclude_tags(args), parse_additional_tags(args), args, interrogator)
                with open(caption_path, 'w') as fp:
                    fp.write(generate_output_string(media_path, tags, args))
                processed_count += 1

                if not args.progress_bar:
                    logger.info("Generated tags",
                               file=str(media_path.name),
                               tag_count=len(tags),
                               output_file=str(caption_path.name))

        except Exception as e:
            error_count += 1
            logger.error("Processing failed",
                        file=str(media_path.name),
                        path=str(media_path),
                        error=str(e),
                        error_type=type(e).__name__)

    # Clean up progress bar
    if _current_progress_bar:
        _current_progress_bar.close()
        _current_progress_bar = None

    logger.info("Batch processing complete",
                processed=processed_count,
                skipped=skipped_count,
                errors=error_count,
                total=len(media_files))


def process_single_file(args, logger, interrogator):
    """Process a single image or video file."""
    file_path = Path(args.file)
    logger.info("Processing single file",
                file=str(file_path.name),
                path=str(file_path),
                model=args.model,
                threshold=args.threshold)
    try:
        if is_video_file(file_path):
            # Process video file
            frames_processed = process_video_file(
                file_path, not args.rawtag, parse_exclude_tags(args),
                parse_additional_tags(args), args, interrogator, logger)

            logger.info("Generated tags for video file",
                       file=str(file_path.name),
                       frames_processed=frames_processed,
                       sample_mode=args.sample)
        else:
            # Process image file
            tags = image_interrogate(file_path, not args.rawtag, parse_exclude_tags(args), parse_additional_tags(args), args, interrogator)
            output = generate_output_string(args.file, tags, args)
            print(output)
            logger.info("Generated tags for single file",
                       file=str(file_path.name),
                       tag_count=len(tags),
                       output_format="json" if args.json else "text")
    except Exception as e:
        logger.error("Single file processing failed",
                    file=str(file_path.name),
                    path=str(file_path),
                    error=str(e),
                    error_type=type(e).__name__)
        sys.exit(1)

def parse_exclude_tags(args) -> set[str]:
    if args.exclude_tags is None:
        return set()

    tags = []
    for str in args.exclude_tags:
        for tag in str.split(','):
            tags.append(tag.strip())

    # reverse escape (nai tag to danbooru tag)
    reverse_escaped_tags = []
    for tag in tags:
        tag = tag.replace(' ', '_').replace(r'\(', '(').replace(r'\)', ')')
        reverse_escaped_tags.append(tag)
    return set([*tags, *reverse_escaped_tags])  # reduce duplicates

def parse_additional_tags(args) -> list[str]:
    if args.additional_tags is None:
        return list()

    tags = []
    for str in args.additional_tags:
        for tag in str.split(','):
            tags.append(tag.strip())
    return list(set(tags))

def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str], additional_tags: list[str], args, interrogator) -> dict[str, float]:
    """
    Predictions from a image path
    """
    im = Image.open(image_path)
    result = interrogator.interrogate(im)

    return AbsInterrogator.postprocess_tags(
        result[1],
        threshold=args.threshold,
        escape_tag=tag_escape,
        replace_underscore=tag_escape,
        exclude_tags=exclude_tags,
        additional_tags=additional_tags)

def explore_media_files(folder_path: Path, recursive: bool) -> Generator[Path, None, None]:
    """
    Explore image and video files by folder path in lexicographic order
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    supported_extensions = image_extensions + video_extensions

    paths = []
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix.lower() in supported_extensions:
            paths.append(path)
        elif recursive and path.is_dir():
            paths.extend(explore_media_files(path, recursive))
    paths.sort()
    yield from paths

def generate_output_string(image_path: str, tags: Dict, args):
    if args.json:
        return json.dumps({"file":str(image_path),"tags":tags})
    else:
        return ', '.join(tags.keys())

def is_video_file(file_path: Path) -> bool:
    """Check if file is a video based on extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return file_path.suffix.lower() in video_extensions

def process_video_file(video_path: Path, tag_escape: bool, exclude_tags: Iterable[str], additional_tags: list[str], args, interrogator, logger) -> int:
    """
    Process a video file by extracting frames and tagging each one.
    Returns the number of frames processed.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine frame sampling strategy
    if args.sample == 'all':
        frame_indices = list(range(total_frames))
    elif args.sample == 'keyframes':
        # For now, implement simple keyframe detection (every 30 frames)
        # TODO: Implement proper scene detection
        frame_indices = list(range(0, total_frames, 30))
    else:
        # Sample every Nth frame
        try:
            step = int(args.sample)
        except ValueError:
            raise ValueError(f"Invalid sample value: {args.sample}. Must be 'all', 'keyframes', or an integer")
        frame_indices = list(range(0, total_frames, step))

    if not args.progress_bar:
        logger.info("Starting video processing",
                   file=str(video_path.name),
                   total_frames=total_frames,
                   sample_mode=args.sample,
                   frames_to_process=len(frame_indices))

    processed_frames = 0

    for frame_idx in frame_indices:
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            logger.warning("Failed to read frame",
                          frame_index=frame_idx,
                          file=str(video_path.name))
            continue

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Interrogate the frame
        result = interrogator.interrogate(image)
        tags = AbsInterrogator.postprocess_tags(
            result[1],
            threshold=args.threshold,
            escape_tag=tag_escape,
            replace_underscore=tag_escape,
            exclude_tags=exclude_tags,
            additional_tags=additional_tags)

        # Generate output file path
        frame_output_path = video_path.parent / f'{video_path.stem}.frame_{frame_idx:06d}{args.ext}'

        # Write tags to file
        with open(frame_output_path, 'w') as fp:
            fp.write(generate_output_string(str(frame_output_path), tags, args))

        processed_frames += 1

        if not args.progress_bar:
            logger.info("Processed video frame",
                       frame_index=frame_idx,
                       file=str(video_path.name),
                       tag_count=len(tags),
                       output_file=str(frame_output_path.name))

    cap.release()

    if not args.progress_bar:
        logger.info("Video processing complete",
                   file=str(video_path.name),
                   frames_processed=processed_frames,
                   total_frames=total_frames)

    return processed_frames



def main():
    """Main entry point for the WD14 tagger."""
    args = parse_args()

    # Initialize structured logging
    logger = setup_logging(args.progress_bar)

    # get interrogator configs
    interrogator = interrogators[args.model]

    if args.cpu:
        interrogator.use_cpu()

    # Set quiet mode if progress bar is enabled
    if args.progress_bar:
        interrogator.set_quiet(True)
        logging.getLogger('tagger.interrogator').setLevel(logging.WARNING)

    # Process based on arguments
    if args.dir:
        process_directory(args, logger, interrogator)
    elif args.file:
        process_single_file(args, logger, interrogator)


if __name__ == "__main__":
    main()
