import sys
import json
import logging
import warnings
from tqdm import tqdm
from typing import Generator, Iterable
from tagger.interrogator.interrogator import AbsInterrogator
from PIL import Image
from pathlib import Path
import argparse

from tagger.interrogators import interrogators

# Configure logging
logging.basicConfig(
    format='%(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Filter CUDA provider warning
warnings.filterwarnings('ignore', message='Specified provider .* is not in available provider names')

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
args = parser.parse_args()

# get interrogator configs
interrogator = interrogators[args.model]

if args.cpu:
    interrogator.use_cpu()

# Set quiet mode if progress bar is enabled
if args.progress_bar:
    interrogator.set_quiet(True)
    logging.getLogger('tagger.interrogator').setLevel(logging.WARNING)

def parse_exclude_tags() -> set[str]:
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

def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]) -> dict[str, float]:
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
        exclude_tags=exclude_tags)

def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
    """
    Explore files by folder path in lexicographic order
    """
    paths = []
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
            paths.append(path)
        elif args.recursive and path.is_dir():
            paths.extend(explore_image_files(path))
    paths.sort()
    yield from paths

def generate_output_string(image_path: str, tags: Dict):
    if args.json:
        return json.dumps({"file":str(image_path),"tags":tags})
    else:
        return ', '.join(tags.keys())

if args.dir:
    root_path = Path(args.dir)
    # First collect all files to process for progress bar and logging
    image_files = list(explore_image_files(root_path))
    
    if not image_files:
        logger.warning(f"No image files found in directory: {root_path}")
        sys.exit(0)
        
    iterator = tqdm(image_files) if args.progress_bar else image_files
    for image_path in iterator:
        caption_path = image_path.parent / f'{image_path.stem}{args.ext}'

        if caption_path.is_file() and not args.overwrite:
            # skip if caption exists
            if not args.progress_bar:
                logger.info(f"Skipping existing file: {image_path}")
            continue

        if not args.progress_bar:
            logger.info(f"Processing: {image_path}")
        try:
            tags = image_interrogate(image_path, not args.rawtag, parse_exclude_tags())
            with open(caption_path, 'w') as fp:
                fp.write(generate_output_string(image_path, tags))
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")

if args.file:
    try:
        tags = image_interrogate(Path(args.file), not args.rawtag, parse_exclude_tags())
        print(generate_output_string(args.file, tags))
    except Exception as e:
        logger.error(f"Failed to process {args.file}: {str(e)}")
        sys.exit(1)

