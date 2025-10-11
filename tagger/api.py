"""
High-level API for WD14 Tagger functionality.

Provides a simple, Pythonic interface for image tagging that abstracts away
the complexity of managing interrogators and vocabularies.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
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
        device: Optional[str] = None,
        deterministic: bool = False,
        quiet: bool = True
    ):
        """Initialize the tagger with a specific model.

        Args:
            model_name: Name of the model to use (see get_available_models())
            device: Device selection ('cpu', 'cuda', 'mps', or None for auto)
            deterministic: Enable deterministic computation if supported
            quiet: Suppress logging output
        """
        if model_name not in interrogators:
            available = list(interrogators.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        self.model_name = model_name
        self.interrogator = interrogators[model_name]
        self.vocabulary: Optional[TagVocabulary] = None

        # Set device preferences
        if device == 'cpu':
            self.interrogator.use_cpu()
        elif device == 'cuda':
            self.interrogator.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        elif device == 'mps':
            self.interrogator.set_providers(['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        elif device is not None:
            raise ValueError(f"Unsupported device: {device}. Use 'cpu', 'cuda', 'mps', or None.")

        self.deterministic = deterministic
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
        image_input: Union[str, Path, Image.Image, np.ndarray],
        threshold: float = 0.35,
        additional_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        sort_by_alphabetical_order: bool = False,
        add_confident_as_weight: bool = False,
        replace_underscore: bool = False,
        escape_tag: bool = False,
        return_logits: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray, List[int]]]:
        """Tag a single image.

        Args:
            image_input: Path to image file, PIL Image object, or numpy array
            threshold: Confidence threshold for tag inclusion
            additional_tags: Tags to always include with confidence 1.0
            exclude_tags: Tags to exclude from results
            sort_by_alphabetical_order: Sort by tag name instead of confidence
            add_confident_as_weight: Format tags as (tag:confidence)
            replace_underscore: Replace underscores with spaces in tag names
            escape_tag: Escape special characters in tag names
            return_logits: Also return raw model logits in vocabulary order

        Returns:
            List of token objects: [{'token_id': int, 'label': str, 'score': float}, ...]
            If return_logits=True: (tokens, logits_array, id_order)
        """
        # Load image if needed
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            if image_input.dtype != np.uint8:
                # Normalize to 0-255 range if not already
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)

            # Handle different array shapes
            if len(image_input.shape) == 3:
                if image_input.shape[2] == 3:  # RGB
                    image = Image.fromarray(image_input, 'RGB')
                elif image_input.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image_input, 'RGBA')
                else:
                    raise ValueError(f"Unsupported number of channels: {image_input.shape[2]}")
            elif len(image_input.shape) == 2:  # Grayscale
                image = Image.fromarray(image_input, 'L')
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")
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
        if return_logits:
            # Convert tag_probs to ordered array matching vocabulary order
            id_order = self.vocabulary.get_all_ids()
            raw_logits = np.zeros(len(id_order))
            for i, token_id in enumerate(id_order):
                tag = self.vocabulary.get_tag_string(token_id)
                if tag and tag in tag_probs:
                    raw_logits[i] = tag_probs[tag]
            return tokens, raw_logits, id_order

        return tokens

    def tag_images(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        threshold: float = 0.35,
        return_logits: bool = False,
        **kwargs
    ) -> Union[List[List[Dict[str, Any]]], Tuple[List[List[Dict[str, Any]]], np.ndarray, List[int]]]:
        """Tag multiple images efficiently.

        Args:
            image_inputs: List of image paths, PIL Image objects, or numpy arrays
            threshold: Confidence threshold for tag inclusion
            return_logits: Also return batched logits with consistent id_order
            **kwargs: Additional arguments passed to tag_image

        Returns:
            List of token lists for each image: [[{'token_id': int, 'label': str, 'score': float}], ...]
            If return_logits=True: (token_lists, logits_batch, id_order)
        """
        results = []
        logits_batch = []
        id_order = None

        for image_input in image_inputs:
            if return_logits:
                tags, logits, id_order = self.tag_image(
                    image_input,
                    threshold=threshold,
                    return_logits=True,
                    **kwargs
                )
                logits_batch.append(logits)
            else:
                tags = self.tag_image(
                    image_input,
                    threshold=threshold,
                    return_logits=False,
                    **kwargs
                )
            results.append(tags)

        if return_logits:
            return results, np.array(logits_batch), id_order
        return results

    def tag_images_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        threshold: float = 0.35,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Legacy alias for tag_images() without logits support.

        Args:
            image_inputs: List of image paths, PIL Image objects, or numpy arrays
            threshold: Confidence threshold for tag inclusion
            **kwargs: Additional arguments passed to tag_image

        Returns:
            List of token lists for each image: [[{'token_id': int, 'label': str, 'score': float}], ...]
        """
        return self.tag_images(image_inputs, threshold=threshold, return_logits=False, **kwargs)

    def iter_tag_images(
        self,
        image_inputs: Iterator[Union[str, Path, Image.Image, np.ndarray]],
        threshold: float = 0.35,
        **kwargs
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream tag results for large image sets to avoid loading all into memory.

        Args:
            image_inputs: Iterator over image paths, PIL Image objects, or numpy arrays
            threshold: Confidence threshold for tag inclusion
            **kwargs: Additional arguments passed to tag_image

        Yields:
            Token list for each image: [{'token_id': int, 'label': str, 'score': float}, ...]
        """
        for image_input in image_inputs:
            result = self.tag_image(
                image_input,
                threshold=threshold,
                return_logits=False,
                **kwargs
            )
            yield result

    def get_vocab(self) -> Dict[int, str]:
        """Get vocabulary as dict[int, str] with stable ordering.

        Returns:
            Dictionary mapping token IDs to tag strings, sorted by ID
        """
        self._ensure_vocabulary()
        return {token_id: tag for token_id, tag in sorted(self.vocabulary.id_to_tag.items())}

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
        """Get comprehensive model information.

        Returns:
            Dictionary with model metadata including vocabulary info and id_order
        """
        self._ensure_vocabulary()
        vocab_metadata = self.vocabulary.get_vocab_metadata()
        id_order = self.vocabulary.get_all_ids()

        return {
            'model': self.model_name,
            'version': vocab_metadata.get('vocab_version', '1.0'),
            'display_name': self.interrogator.name,
            'is_loaded': self.is_loaded,
            'providers': self.interrogator.providers,
            'vocab_size': vocab_metadata['vocab_size'],
            'vocab_hash': vocab_metadata['vocab_hash'],
            'id_order': id_order,
            'deterministic': self.deterministic
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