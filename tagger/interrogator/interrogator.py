import re
import sys
import logging
import structlog

import pandas as pd

from typing import Iterable, Tuple, List, Dict, Optional, Union
from PIL import Image
from pathlib import Path

from onnxruntime import InferenceSession, get_available_providers

logger = structlog.get_logger(__name__)
tag_escape_pattern = re.compile(r'([\\()])')

class AbsInterrogator:
    model: InferenceSession | None
    tags: pd.DataFrame | None
    vocabulary: Optional['TagVocabulary']
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: Iterable[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False,
        return_tokens=False,
        vocabulary=None
    ) -> Union[Dict[str, float], Dict[int, float]]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        # Convert to token IDs if requested
        if return_tokens:
            if vocabulary is None:
                raise ValueError("vocabulary parameter is required when return_tokens=True")
            return vocabulary.tags_to_token_dict(tags, skip_unknown=True)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.quiet = False
        self.vocabulary = None

    def set_quiet(self, quiet: bool) -> None:
        self.quiet = quiet

    def set_vocabulary(self, vocabulary: 'TagVocabulary') -> None:
        """Set the vocabulary for this interrogator."""
        self.vocabulary = vocabulary

    def build_vocabulary_from_tags(self) -> 'TagVocabulary':
        """Build vocabulary from this interrogator's tag list."""
        from tagger.vocabulary import TagVocabulary

        if not hasattr(self, 'tags') or self.tags is None:
            self.load()

        vocab = TagVocabulary()

        # Handle standard pandas DataFrame format (WD14, MLDanbooru models)
        if hasattr(self, 'tags') and self.tags is not None and hasattr(self.tags, 'columns') and 'name' in self.tags.columns:
            tags = self.tags['name'].tolist()
            vocab.add_tags_from_list([tag for tag in tags if isinstance(tag, str)])

        # Handle Camie Tagger metadata format
        elif hasattr(self, 'metadata') and self.metadata is not None and 'idx_to_tag' in self.metadata:
            # Get all tags from the idx_to_tag mapping
            tags = list(self.metadata['idx_to_tag'].values())
            vocab.add_tags_from_list([tag for tag in tags if isinstance(tag, str)])

        return vocab

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            if not self.quiet:
                logger.info("Model unloaded", model=self.name)

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def use_cpu(self) -> None:
        """Force CPU-only execution."""
        self.providers = ['CPUExecutionProvider']
        logger.info("Forcing CPU execution", model=self.name)

    def get_available_providers(self) -> List[str]:
        """Get list of available execution providers."""
        return get_available_providers()

    def set_providers(self, providers: List[str]) -> None:
        """Set specific execution providers."""
        self.providers = providers
        #print(f'Set custom providers for {self.name}: {providers}', file=sys.stderr)

    def get_optimal_provider(self) -> List[str]:
        """Get the optimal provider based on system capabilities.
        
        Returns a list of providers in order of preference:
        - CoreMLExecutionProvider (if on Apple Silicon)
        - CUDAExecutionProvider (if NVIDIA GPU available)
        - CPUExecutionProvider (always available as fallback)
        """
        available = self.get_available_providers()
        
        # Start with most optimal providers first
        optimal_order = [
            'CoreMLExecutionProvider',  # Best for Apple Silicon
            'CUDAExecutionProvider',    # Best for NVIDIA GPUs
            'CPUExecutionProvider'      # Fallback
        ]
        
        # Return list of available providers in optimal order
        selected = [p for p in optimal_order if p in available]
        #print(f'Selected optimal providers for {self.name}: {selected}', file=sys.stderr)
        return selected

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()
