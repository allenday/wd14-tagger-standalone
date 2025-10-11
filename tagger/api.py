"""
High-level API for WD14 Tagger functionality.

Provides a simple, Pythonic interface for image tagging that abstracts away
the complexity of managing interrogators and vocabularies.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

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
        use_cpu: bool = False,
        quiet: bool = True
    ):
        """Initialize the tagger with a specific model.

        Args:
            model_name: Name of the model to use (see get_available_models())
            use_cpu: Force CPU-only execution
            quiet: Suppress logging output
        """
        if model_name not in interrogators:
            available = list(interrogators.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        self.model_name = model_name
        self.interrogator = interrogators[model_name]
        self.vocabulary: Optional[TagVocabulary] = None

        if use_cpu:
            self.interrogator.use_cpu()

        self.interrogator.set_quiet(quiet)

        # Always build vocabulary for token support
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
        additional_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        sort_by_alphabetical_order: bool = False,
        add_confident_as_weight: bool = False,
        replace_underscore: bool = False,
        escape_tag: bool = False,
        return_raw_logits: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
        """Tag a single image.

        Args:
            image_input: Path to image file or PIL Image object
            threshold: Confidence threshold for tag inclusion
            additional_tags: Tags to always include with confidence 1.0
            exclude_tags: Tags to exclude from results
            sort_by_alphabetical_order: Sort by tag name instead of confidence
            add_confident_as_weight: Format tags as (tag:confidence)
            replace_underscore: Replace underscores with spaces in tag names
            escape_tag: Escape special characters in tag names
            return_raw_logits: Also return raw model logits in fixed order

        Returns:
            List of token objects: [{'token_id': int, 'label': str, 'score': float}, ...]
            If return_raw_logits=True: (token_list, raw_logits_array)
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

        # Process tags using the traditional method first
        processed_tags = AbsInterrogator.postprocess_tags(
            tag_probs,
            threshold=threshold,
            additional_tags=additional_tags,
            exclude_tags=exclude_tags,
            sort_by_alphabetical_order=sort_by_alphabetical_order,
            add_confident_as_weight=add_confident_as_weight,
            replace_underscore=replace_underscore,
            escape_tag=escape_tag,
            return_tokens=False,
            vocabulary=None
        )

        # Convert to structured token format
        tokens = []
        for tag, score in processed_tags.items():
            token_id = self.vocabulary.get_tag_id(tag)
            if token_id is not None:  # Only include tags that have token IDs
                tokens.append({
                    'token_id': token_id,
                    'label': tag,
                    'score': float(score)
                })

        # Sort by score (descending) if not alphabetical
        if not sort_by_alphabetical_order:
            tokens.sort(key=lambda x: x['score'], reverse=True)
        else:
            tokens.sort(key=lambda x: x['label'])

        # Return raw logits if requested
        if return_raw_logits:
            # Convert tag_probs to ordered array matching vocabulary order
            vocab_size = len(self.vocabulary.get_all_tags())
            raw_logits = np.zeros(vocab_size)
            for tag, prob in tag_probs.items():
                token_id = self.vocabulary.get_tag_id(tag)
                if token_id is not None and token_id < vocab_size:
                    raw_logits[token_id] = prob
            return tokens, raw_logits

        return tokens

    def tag_images_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        threshold: float = 0.35,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Tag multiple images efficiently.

        Args:
            image_inputs: List of image paths or PIL Image objects
            threshold: Confidence threshold for tag inclusion
            **kwargs: Additional arguments passed to tag_image

        Returns:
            List of token lists for each image: [[{'token_id': int, 'label': str, 'score': float}], ...]
        """
        results = []
        for image_input in image_inputs:
            tags = self.tag_image(
                image_input,
                threshold=threshold,
                **kwargs
            )
            # Handle case where raw_logits might be returned
            if isinstance(tags, tuple):
                tags = tags[0]  # Take just the token list
            results.append(tags)
        return results

    def get_vocabulary(self) -> TagVocabulary:
        """Get the vocabulary for this model.

        Returns:
            TagVocabulary object for converting between tags and tokens
        """
        self._ensure_vocabulary()
        return self.vocabulary

    def dump_vocab(self) -> List[Dict[str, Any]]:
        """Export full vocabulary as structured data.

        Returns:
            List of dictionaries with token_id, label for each vocabulary entry
        """
        self._ensure_vocabulary()
        return self.vocabulary.dump_vocab()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model metadata including vocabulary info
        """
        self._ensure_vocabulary()
        vocab_metadata = self.vocabulary.get_vocab_metadata()

        return {
            'name': self.model_name,
            'display_name': self.interrogator.name,
            'is_loaded': self.is_loaded,
            'providers': self.interrogator.providers,
            **vocab_metadata  # vocab_size, vocab_version, vocab_hash
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
        vocab_size = len(self.vocabulary.get_all_tags()) if self.vocabulary else "unknown"
        return f"WD14Tagger(model='{self.model_name}', {status}, vocab_size={vocab_size})"