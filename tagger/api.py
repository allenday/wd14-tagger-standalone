"""
High-level API for WD14 Tagger functionality.

Provides a simple, Pythonic interface for image tagging that abstracts away
the complexity of managing interrogators and vocabularies.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from PIL import Image

from .interrogators import interrogators
from .interrogator.interrogator import AbsInterrogator
from .vocabulary import TagVocabulary


class WD14Tagger:
    """High-level interface for WD14 image tagging.

    This class provides a simple API for loading models and tagging images,
    with support for both string tags and integer token outputs.

    Example:
        >>> tagger = WD14Tagger('wd14-convnextv2.v1')
        >>> tags = tagger.tag_image('image.jpg')
        >>> print(tags)
        {'1girl': 0.95, 'solo': 0.89, ...}
    """

    def __init__(
        self,
        model_name: str = 'wd14-convnextv2.v1',
        enable_tokens: bool = False,
        use_cpu: bool = False,
        quiet: bool = True
    ):
        """Initialize the tagger with a specific model.

        Args:
            model_name: Name of the model to use (see get_available_models())
            enable_tokens: Whether to build vocabulary for token output
            use_cpu: Force CPU-only execution
            quiet: Suppress logging output
        """
        if model_name not in interrogators:
            available = list(interrogators.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        self.model_name = model_name
        self.interrogator = interrogators[model_name]
        self.enable_tokens = enable_tokens
        self.vocabulary: Optional[TagVocabulary] = None

        if use_cpu:
            self.interrogator.use_cpu()

        self.interrogator.set_quiet(quiet)

        # Pre-build vocabulary if tokens are enabled
        if enable_tokens:
            self._ensure_vocabulary()

    def _ensure_vocabulary(self):
        """Ensure vocabulary is built for token output."""
        if self.vocabulary is None:
            self.vocabulary = self.interrogator.build_vocabulary_from_tags()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return hasattr(self.interrogator, 'model') and self.interrogator.model is not None

    def load_model(self):
        """Explicitly load the model into memory."""
        if not self.is_loaded:
            self.interrogator.load()

    def unload_model(self):
        """Unload the model from memory to free resources."""
        self.interrogator.unload()

    def tag_image(
        self,
        image_input: Union[str, Path, Image.Image],
        threshold: float = 0.35,
        return_tokens: bool = False,
        additional_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        sort_by_alphabetical_order: bool = False,
        add_confident_as_weight: bool = False,
        replace_underscore: bool = False,
        escape_tag: bool = False
    ) -> Union[Dict[str, float], Dict[int, float]]:
        """Tag a single image.

        Args:
            image_input: Path to image file or PIL Image object
            threshold: Confidence threshold for tag inclusion
            return_tokens: Return token IDs instead of tag strings
            additional_tags: Tags to always include with confidence 1.0
            exclude_tags: Tags to exclude from results
            sort_by_alphabetical_order: Sort by tag name instead of confidence
            add_confident_as_weight: Format tags as (tag:confidence)
            replace_underscore: Replace underscores with spaces in tag names
            escape_tag: Escape special characters in tag names

        Returns:
            Dictionary mapping tags/tokens to confidence scores
        """
        # Load image if needed
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input

        # Ensure model is loaded
        self.load_model()

        # Get raw predictions
        rating_probs, tag_probs = self.interrogator.interrogate(image)

        # Prepare parameters
        additional_tags = additional_tags or []
        exclude_tags = set(exclude_tags or [])

        # Handle token output
        if return_tokens:
            if not self.enable_tokens:
                raise ValueError("Token output requires enable_tokens=True during initialization")
            self._ensure_vocabulary()
            vocabulary = self.vocabulary
        else:
            vocabulary = None

        # Post-process tags
        return AbsInterrogator.postprocess_tags(
            tag_probs,
            threshold=threshold,
            additional_tags=additional_tags,
            exclude_tags=exclude_tags,
            sort_by_alphabetical_order=sort_by_alphabetical_order,
            add_confident_as_weight=add_confident_as_weight,
            replace_underscore=replace_underscore,
            escape_tag=escape_tag,
            return_tokens=return_tokens,
            vocabulary=vocabulary
        )

    def tag_images_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        threshold: float = 0.35,
        return_tokens: bool = False,
        **kwargs
    ) -> List[Union[Dict[str, float], Dict[int, float]]]:
        """Tag multiple images efficiently.

        Args:
            image_inputs: List of image paths or PIL Image objects
            threshold: Confidence threshold for tag inclusion
            return_tokens: Return token IDs instead of tag strings
            **kwargs: Additional arguments passed to tag_image

        Returns:
            List of tag/token dictionaries for each image
        """
        results = []
        for image_input in image_inputs:
            tags = self.tag_image(
                image_input,
                threshold=threshold,
                return_tokens=return_tokens,
                **kwargs
            )
            results.append(tags)
        return results

    def get_vocabulary(self) -> TagVocabulary:
        """Get the vocabulary for this model.

        Returns:
            TagVocabulary object for converting between tags and tokens

        Raises:
            RuntimeError: If tokens are not enabled
        """
        if not self.enable_tokens:
            raise RuntimeError("Vocabulary access requires enable_tokens=True")
        self._ensure_vocabulary()
        return self.vocabulary

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.model_name,
            'display_name': self.interrogator.name,
            'is_loaded': self.is_loaded,
            'tokens_enabled': self.enable_tokens,
            'vocab_size': len(self.vocabulary.get_all_tags()) if self.vocabulary else None,
            'providers': self.interrogator.providers
        }

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload_model()

    def __repr__(self):
        """String representation."""
        status = "loaded" if self.is_loaded else "unloaded"
        tokens = "tokens enabled" if self.enable_tokens else "tags only"
        return f"WD14Tagger(model='{self.model_name}', {status}, {tokens})"