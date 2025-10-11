"""
WD14 Tagger Standalone - A library for image tagging using WD14 models.

This package provides both CLI and programmatic access to WD14 image tagging models.
Supports multiple model architectures and can output both tag strings and token IDs.

Example usage:
    Basic tagging:
        >>> from tagger import WD14Tagger
        >>> tagger = WD14Tagger('wd14-convnextv2.v1')
        >>> tags = tagger.tag_image('path/to/image.jpg')
        >>> print(tags)
        {'1girl': 0.95, 'solo': 0.89, ...}

    With token output:
        >>> tagger = WD14Tagger('wd14-convnextv2.v1', enable_tokens=True)
        >>> tokens = tagger.tag_image('path/to/image.jpg', return_tokens=True)
        >>> print(tokens)
        {28: 0.95, 5: 0.89, ...}
"""

from .interrogators import interrogators
from .interrogator.interrogator import AbsInterrogator
from .vocabulary import TagVocabulary
from .api import WD14Tagger

__version__ = "0.1.6"
__author__ = "corkborg, Allen Day"
__email__ = "corkborg@users.noreply.github.com"

# Main public API
__all__ = [
    "WD14Tagger",
    "interrogators",
    "AbsInterrogator",
    "TagVocabulary",
    "get_available_models",
    "tag_image",
    "tag_images_batch",
    "dump_vocab",
]

def get_available_models():
    """Get list of available model names.

    Returns:
        List[str]: Available model identifiers
    """
    return list(interrogators.keys())

def tag_image(image_path, model_name='wd14-convnextv2.v1', threshold=0.35, **kwargs):
    """Quick function to tag a single image.

    Args:
        image_path (str): Path to image file
        model_name (str): Model identifier (default: 'wd14-convnextv2.v1')
        threshold (float): Confidence threshold (default: 0.35)
        **kwargs: Additional arguments passed to tag_image

    Returns:
        List[Dict[str, Any]]: List of token objects: [{'token_id': int, 'label': str, 'score': float}, ...]
    """
    tagger = WD14Tagger(model_name)
    return tagger.tag_image(image_path, threshold=threshold, **kwargs)

def tag_images_batch(image_paths, model_name='wd14-convnextv2.v1', threshold=0.35, **kwargs):
    """Tag multiple images efficiently.

    Args:
        image_paths (List[str]): List of image file paths
        model_name (str): Model identifier (default: 'wd14-convnextv2.v1')
        threshold (float): Confidence threshold (default: 0.35)
        **kwargs: Additional arguments passed to tag_image

    Returns:
        List[List[Dict[str, Any]]]: List of token lists for each image
    """
    tagger = WD14Tagger(model_name)
    return tagger.tag_images_batch(image_paths, threshold=threshold, **kwargs)

def dump_vocab(model_name='wd14-convnextv2.v1'):
    """Dump vocabulary for a specific model.

    Args:
        model_name (str): Model identifier (default: 'wd14-convnextv2.v1')

    Returns:
        List[Dict[str, Any]]: List of token objects: [{'token_id': int, 'label': str}, ...]
    """
    tagger = WD14Tagger(model_name)
    return tagger.dump_vocab()