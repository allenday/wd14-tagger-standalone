#!/usr/bin/env python3
"""
Download and cache model files from HuggingFace before the Docker build process.
This makes the build more reliable and avoids rate limits during container startup.
"""
import os
import sys
import json
import shutil
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)

def setup_model_cache(output_dir="hf_cache", force=False):
    """
    Download model files from HuggingFace and store them in the correct cache structure.
    This directory can then be copied directly into the Docker image at the standard cache location.
    """
    # Model configuration - repo_id, filename
    models = [
        {
            "repo_id": "Camais03/camie-tagger",
            "files": [
                "model_initial.onnx",
                "tag_mapping_v1.json"
            ]
        }
    ]
    
    # Create base cache directory that mimics huggingface_hub structure
    # The Docker image will COPY this entire structure to /root/.cache/huggingface/
    output_path = Path(output_dir).absolute()
    hub_path = output_path / "hub"
    
    if output_path.exists() and force:
        logger.warning(f"Removing existing cache directory: {output_path}")
        shutil.rmtree(output_path)
    
    hub_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Model cache directory: {output_path}")
    
    # Cache location structure:
    # hub/
    #   models--{org}--{model}/
    #     snapshots/
    #       latest/
    #     blobs/
    #       {file_hash}
    
    # Download each model
    files_cached = []
    for model in models:
        repo_id = model["repo_id"]
        # Convert repo_id to huggingface's directory format
        repo_dir_name = f"models--{repo_id.replace('/', '--')}"
        
        # Create directory structure
        repo_path = hub_path / repo_dir_name
        snapshot_path = repo_path / "snapshots" / "latest"
        blobs_path = repo_path / "blobs"
        
        snapshot_path.mkdir(exist_ok=True, parents=True)
        blobs_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Processing model: {repo_id}")
        
        # Download each file
        for file in model["files"]:
            # First try to download to huggingface's default cache
            logger.info(f"Downloading {file}")
            success = False
            
            for attempt in range(3):
                try:
                    logger.info(f"Download attempt {attempt+1}/3")
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file
                    )
                    logger.info(f"Downloaded to {file_path}")
                    
                    # The file will be saved to the default cache location
                    # We'll copy it to our output directory with the same structure
                    
                    # Get the hashed filename that huggingface uses
                    blob_path = Path(file_path)
                    blob_name = blob_path.name
                    
                    # Copy to our cache structure
                    target_path = blobs_path / blob_name
                    logger.info(f"Copying to {target_path}")
                    shutil.copy(file_path, target_path)
                    
                    # Also create a reference file in snapshots/latest
                    with open(snapshot_path / file, "w") as f:
                        f.write(f"../blobs/{blob_name}")
                    
                    files_cached.append(f"{repo_id}/{file} -> {blob_name}")
                    success = True
                    break
                    
                except Exception as e:
                    logger.error(f"Error downloading {file}: {str(e)}")
            
            if not success:
                if file.endswith(".json"):
                    # Create fallback for tag mapping
                    logger.warning(f"Creating fallback for {file}")
                    tag_mapping = {
                        'idx_to_tag': {str(i): f'tag_{i}' for i in range(100)},
                        'tag_to_category': {f'tag_{i}': 'general' for i in range(100)}
                    }
                    
                    # Generate a consistent filename for the fallback
                    fallback_name = "fallback_tag_mapping_v1.json"
                    target_path = blobs_path / fallback_name
                    
                    with open(target_path, 'w') as f:
                        json.dump(tag_mapping, f)
                    
                    # Create reference in snapshots
                    with open(snapshot_path / file, "w") as f:
                        f.write(f"../blobs/{fallback_name}")
                    
                    logger.info(f"Created fallback at {target_path}")
                    files_cached.append(f"{repo_id}/{file} -> {fallback_name} (FALLBACK)")
                else:
                    logger.error(f"Failed to download {file} after 3 attempts")
                    logger.error(f"Please manually download it from huggingface.co/{repo_id}/resolve/main/{file}")
                    logger.error(f"and place it in {blobs_path}")
    
    # Create metadata file to track what was cached
    metadata = {
        "models": [m["repo_id"] for m in models],
        "files_cached": files_cached,
        "usage": "Copy the entire 'hub' directory to /root/.cache/huggingface/ in your Docker image"
    }
    
    with open(output_path / "cache_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model cache complete. Files cached:")
    for file in files_cached:
        logger.info(f"- {file}")
    
    return output_path

def main():
    """Parse arguments and run the model cache setup."""
    parser = argparse.ArgumentParser(
        description="Download and cache model files for Docker build",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", default="models_cache", 
                        help="Directory to store cached models")
    parser.add_argument("--force", action="store_true", 
                        help="Force re-download of existing files")
    
    args = parser.parse_args()
    
    try:
        cache_dir = setup_model_cache(args.output_dir, args.force)
        print(f"\nSuccess! Model files cached in: {cache_dir}")
        print("Use this directory in your Dockerfile with COPY.")
    except Exception as e:
        logger.error(f"Error setting up model cache: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()