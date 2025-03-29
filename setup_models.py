#!/usr/bin/env python3
"""
Setup script to download or generate model files needed for the tagger.
"""

import os
import json
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_huggingface_model():
    """Download model from Hugging Face or create fallbacks."""
    try:
        from huggingface_hub import hf_hub_download
        
        # Setup paths
        cache_dir = Path.home() / '.cache/huggingface'
        blobs_dir = cache_dir / 'hub/models--Camais03--camie-tagger/blobs'
        snapshots_dir = cache_dir / 'hub/models--Camais03--camie-tagger/snapshots/latest'
        
        # Create directories
        os.makedirs(blobs_dir, exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Try to download ONNX model
        logger.info("Downloading ONNX model from Hugging Face...")
        try:
            # Use a more reliable approach with token handling and retries
            from huggingface_hub import login
            import os
            
            # Check if token environment variable is set
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")
                login(token=hf_token)
            
            # Try to download with retries
            for attempt in range(3):
                try:
                    logger.info(f"Download attempt {attempt+1}/3")
                    model_path = hf_hub_download(
                        repo_id='Camais03/camie-tagger', 
                        filename='model_initial.onnx', 
                        cache_dir=cache_dir,
                        force_download=attempt > 0  # Force on retry
                    )
                    logger.info(f"Successfully downloaded model to {model_path}")
                    break
                except Exception as e:
                    logger.error(f"Error on attempt {attempt+1}: {str(e)}")
                    if attempt == 2:  # Last attempt
                        raise
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            logger.warning("Model download failed, function will attempt to download at runtime")
        
        # Create tag mapping either by downloading or fallback
        mapping_file = blobs_dir / 'c7dc4e38696a812e593916e3f2e51b92f687f8ea'
        
        if not mapping_file.exists():
            logger.info("Creating tag mapping...")
            try:
                # First attempt with hf_hub_download
                try:
                    for attempt in range(3):
                        try:
                            logger.info(f"Tag mapping download attempt {attempt+1}/3 using HF Hub")
                            tags_path = hf_hub_download(
                                repo_id='Camais03/camie-tagger',
                                filename='tag_mapping_v1.json',
                                cache_dir=cache_dir,
                                force_download=attempt > 0
                            )
                            logger.info(f"Successfully downloaded tag mapping to {tags_path}")
                            # Don't need to manually create mapping_file since hf_hub_download stores it
                            break
                        except Exception as e:
                            logger.error(f"Error on attempt {attempt+1}: {str(e)}")
                            if attempt == 2:  # Last attempt with hub
                                raise
                except Exception as hub_error:
                    # Fallback to direct download
                    logger.warning(f"HF Hub download failed: {hub_error}, trying direct download")
                    import requests
                    
                    # Try to download tag mapping directly
                    url = "https://huggingface.co/Camais03/camie-tagger/resolve/main/tag_mapping_v1.json"
                    response = requests.get(url)
                    if response.status_code == 200:
                        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
                        with open(mapping_file, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Downloaded tag mapping to {mapping_file}")
                    else:
                        raise Exception(f"Failed to download tag mapping: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading tag mapping: {str(e)}")
                logger.info("Creating fallback tag mapping...")
                
                # Create fallback mapping
                tag_mapping = {
                    'idx_to_tag': {str(i): f'tag_{i}' for i in range(100)},
                    'tag_to_category': {f'tag_{i}': 'general' for i in range(100)}
                }
                
                os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
                with open(mapping_file, 'w') as f:
                    json.dump(tag_mapping, f)
                logger.info(f"Created fallback tag mapping at {mapping_file}")
        else:
            logger.info(f"Tag mapping already exists at {mapping_file}")
            
        return True
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = setup_huggingface_model()
    sys.exit(0 if success else 1)