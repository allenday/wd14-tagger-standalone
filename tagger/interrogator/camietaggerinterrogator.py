"""
Interrogator for the Camie-Tagger model.
reference:
https://huggingface.co/Camais03/camie-tagger/blob/main/onnx_inference.py
"""
import sys
import json
import os
import logging

from typing import cast

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from tagger.interrogator import AbsInterrogator

logger = logging.getLogger(__name__)

class CamieTaggerInterrogator(AbsInterrogator):
    repo_id: str
    model_path: str
    tags_path: str

    def __init__(
        self,
        name: str,
        repo_id: str,
        model_path: str,
        tags_path='metadata.json',
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.repo_id = repo_id
        self.tags = None
        self.model = None

    def download(self) -> tuple[str, str]:
        print(f"Loading {self.name} model file from {self.repo_id}", file=sys.stderr)
        
        # Method 1: Try to find files in the blobs directory first (most reliable)
        model_path, tags_path = self._try_load_from_blobs()
        if model_path and tags_path:
            return model_path, tags_path
            
        # Method 2: Try using the HuggingFace Hub API
        model_path, tags_path = self._try_huggingface_api()
        if model_path and tags_path:
            return model_path, tags_path
            
        # If we got here, all methods failed
        raise RuntimeError("Failed to load model files through any method")
    
    def _try_load_from_blobs(self) -> tuple[str, str]:
        """Try to load directly from the HuggingFace cache."""
        # First try to find the metadata file in the snapshots directory
        # This is where the real model_initial_metadata.json would be
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = os.path.join(cache_dir, f"models--{self.repo_id.replace('/', '--')}")
        
        # First check in snapshots for the metadata file (highest priority)
        snapshot_dir = os.path.join(model_dir, "snapshots")
        if os.path.exists(snapshot_dir):
            print(f"Found snapshots directory: {snapshot_dir}", file=sys.stderr)
            
            # Look through all snapshot directories
            for snap_version in os.listdir(snapshot_dir):
                snap_path = os.path.join(snapshot_dir, snap_version)
                if os.path.isdir(snap_path):
                    metadata_path = os.path.join(snap_path, self.tags_path)
                    if os.path.exists(metadata_path):
                        print(f"Found metadata in snapshot: {metadata_path}", file=sys.stderr)
                        
                        # Now find the model file in the same directory
                        model_file = os.path.join(snap_path, self.model_path)
                        if os.path.exists(model_file):
                            print(f"Found model in snapshot: {model_file}", file=sys.stderr)
                            return model_file, metadata_path
        
        # If not found in snapshots, check blobs directory
        blobs_dir = os.path.join(model_dir, "blobs")
        if os.path.exists(blobs_dir):
            print(f"Found blobs directory: {blobs_dir}", file=sys.stderr)
            
            # Look for files in the blobs directory
            blob_files = os.listdir(blobs_dir)
            
            # Find model file
            model_blob = None
            for f in blob_files:
                if f.endswith('.onnx') or self.model_path in f:
                    model_blob = os.path.join(blobs_dir, f)
                    if os.path.getsize(model_blob) > 10000:  # Must be substantial
                        print(f"Found model file in blobs: {model_blob}", file=sys.stderr)
                        break
            
            # Find metadata file - specifically look for model_initial_metadata.json
            tags_blob = None
            for f in blob_files:
                if f == "model_initial_metadata.json" or self.tags_path in f:
                    tags_blob = os.path.join(blobs_dir, f)
                    print(f"Found tags file in blobs: {tags_blob}", file=sys.stderr)
                    break
            
            # Fall back to any JSON file if needed
            if tags_blob is None:
                for f in blob_files:
                    if f.endswith('.json'):
                        tags_blob = os.path.join(blobs_dir, f)
                        print(f"Found fallback tags file in blobs: {tags_blob}", file=sys.stderr)
                        break
            
            if model_blob and tags_blob:
                return model_blob, tags_blob
                
        print("Could not find files in cache directory", file=sys.stderr)
        return None, None
    
    def _try_huggingface_api(self) -> tuple[str, str]:
        """Try downloading via HuggingFace Hub API."""
        try:
            print(f"Downloading using HuggingFace API", file=sys.stderr)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_path
            )
            tags_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.tags_path
            )
            print(f"Successfully downloaded model files to {model_path}", file=sys.stderr)
            return model_path, tags_path
        except Exception as e:
            print(f"ERROR: HuggingFace download failed: {str(e)}", file=sys.stderr)
            return None, None

    def load(self) -> None:
        model_path, tags_path = self.download()

        self.model = InferenceSession(model_path,
                                        providers=self.providers)
        print(f'Loaded {self.name} model from {model_path}', file=sys.stderr)

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.metadata = json.load(filen)
            
        # Check if we are using a fallback mapping
        is_fallback = all(tag.startswith("tag_") for tag in list(self.metadata['idx_to_tag'].values())[:100])
        if is_fallback:
            print(f"Detected fallback tag mapping with {len(self.metadata['idx_to_tag'])} entries", file=sys.stderr)
            
            # Ensure we have a reasonable number of tags
            if len(self.metadata['idx_to_tag']) < 1000:
                print(f"Fallback mapping is too small, we'll continue but tags may display as unknown-X", file=sys.stderr)

    def interrogate(
        self,
        image: Image.Image
    ) -> tuple[
        dict[str, float],  # rating confidents
        dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            self.load()
        if self.model is None:
            raise Exception("Model not loading.")

        img_tensor = preprocess_image(image)
        img_numpy = img_tensor.unsqueeze(0).numpy()

        input_ = self.model.get_inputs()[0]

        # evaluate model
        outputs = self.model.run(None, {input_.name: img_numpy})

        # Process outputs
        initial_probs: np.ndarray = 1.0 / (1.0 + np.exp(-outputs[0]))  # Apply sigmoid
        refined_probs: np.ndarray = 1.0 / (1.0 + np.exp(-outputs[1])) if len(outputs) > 1 else initial_probs

        # Get top tags
        indices = np.atleast_1d(refined_probs[0]).nonzero()[0]

        # Group by category
        tags_by_category = {}
        for idx in indices:
            idx_str = str(idx)
            tag_name = self.metadata['idx_to_tag'].get(idx_str, f"unknown-{idx}")
            category = self.metadata['tag_to_category'].get(tag_name, "general")

            if category not in tags_by_category:
                tags_by_category[category] = []

            prob = float(refined_probs[0, idx])
            tags_by_category[category].append((tag_name, prob))

        # 'year', 'rating', 'general', 'character', 'copyright', 'artist', 'meta'
        # Handle case where 'rating' category might be missing
        rating_tags = {}
        if 'rating' in tags_by_category:
            rating_tags = dict(tags_by_category['rating'])
            
        # Ensure 'general' category exists, default to empty if missing
        general_tags = {}
        if 'general' in tags_by_category:
            general_tags = dict(tags_by_category['general'])
            
        return rating_tags, general_tags

def preprocess_image(img: Image.Image, image_size=512) -> torch.Tensor:
    """Process an image for inference"""

    # Convert RGBA or Palette images to RGB
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # Get original dimensions
    width, height = img.size
    aspect_ratio = width / height

    # Calculate new dimensions to maintain aspect ratio
    if aspect_ratio > 1:
        new_width = image_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = image_size
        new_width = int(new_height * aspect_ratio)

    # Resize with LANCZOS filter
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with padding
    new_image = Image.new('RGB', (image_size, image_size), (0, 0, 0))
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2
    new_image.paste(img, (paste_x, paste_y))

    import torchvision.transforms as transforms

    # Initialize transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply transforms
    img_tensor= transform(new_image)
    img_tensor = cast(torch.Tensor, img_tensor)
    return img_tensor
