import json
import hashlib
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import pandas as pd


class TagVocabulary:
    """
    Manages mapping between tag strings and integer token IDs for tagger models.

    The vocabulary creates consistent integer IDs for tags across different models
    and provides methods to convert between string tags and token IDs.
    """

    def __init__(self, vocab_path: Optional[Path] = None):
        """Initialize vocabulary, optionally loading from existing file."""
        self.tag_to_id: Dict[str, int] = {}
        self.id_to_tag: Dict[int, str] = {}
        self.next_id = 0
        self.vocab_path = vocab_path

        if vocab_path and vocab_path.exists():
            self.load_vocabulary(vocab_path)

    def add_tags_from_csv(self, csv_path: Path, tag_column: str = 'name') -> None:
        """Add tags from a CSV file (like selected_tags.csv from models)."""
        df = pd.read_csv(csv_path)
        if tag_column not in df.columns:
            raise ValueError(f"Column '{tag_column}' not found in CSV file")

        tags = df[tag_column].tolist()
        for tag in tags:
            if isinstance(tag, str):  # Skip NaN values
                self.add_tag(tag)

    def add_tags_from_list(self, tags: List[str]) -> None:
        """Add multiple tags from a list."""
        for tag in tags:
            self.add_tag(tag)

    def add_tag(self, tag: str) -> int:
        """Add a single tag and return its ID. Returns existing ID if tag already exists."""
        if tag in self.tag_to_id:
            return self.tag_to_id[tag]

        tag_id = self.next_id
        self.tag_to_id[tag] = tag_id
        self.id_to_tag[tag_id] = tag
        self.next_id += 1
        return tag_id

    def get_tag_id(self, tag: str) -> Optional[int]:
        """Get the ID for a tag string, or None if not in vocabulary."""
        return self.tag_to_id.get(tag)

    def get_tag_string(self, tag_id: int) -> Optional[str]:
        """Get the tag string for an ID, or None if not in vocabulary."""
        return self.id_to_tag.get(tag_id)

    def get_tag_ids(self, tags: List[str]) -> List[Optional[int]]:
        """Get IDs for multiple tags. Returns None for unknown tags."""
        return [self.get_tag_id(tag) for tag in tags]

    def get_tag_strings(self, tag_ids: List[int]) -> List[Optional[str]]:
        """Get tag strings for multiple IDs. Returns None for unknown IDs."""
        return [self.get_tag_string(tag_id) for tag_id in tag_ids]

    def tags_to_token_dict(self, tag_dict: Dict[str, float], skip_unknown: bool = True) -> Dict[int, float]:
        """
        Convert a tag confidence dict to token ID dict.

        Args:
            tag_dict: Dictionary mapping tag strings to confidence scores
            skip_unknown: If True, skip tags not in vocabulary. If False, raise error.

        Returns:
            Dictionary mapping token IDs to confidence scores
        """
        token_dict = {}
        for tag, confidence in tag_dict.items():
            tag_id = self.get_tag_id(tag)
            if tag_id is not None:
                token_dict[tag_id] = confidence
            elif not skip_unknown:
                raise ValueError(f"Tag '{tag}' not found in vocabulary")

        return token_dict

    def tokens_to_tag_dict(self, token_dict: Dict[int, float], skip_unknown: bool = True) -> Dict[str, float]:
        """
        Convert a token ID dict to tag confidence dict.

        Args:
            token_dict: Dictionary mapping token IDs to confidence scores
            skip_unknown: If True, skip IDs not in vocabulary. If False, raise error.

        Returns:
            Dictionary mapping tag strings to confidence scores
        """
        tag_dict = {}
        for token_id, confidence in token_dict.items():
            tag = self.get_tag_string(token_id)
            if tag is not None:
                tag_dict[tag] = confidence
            elif not skip_unknown:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")

        return tag_dict

    def save_vocabulary(self, path: Path) -> None:
        """Save vocabulary to JSON file."""
        vocab_data = {
            'tag_to_id': self.tag_to_id,
            'id_to_tag': {str(k): v for k, v in self.id_to_tag.items()},  # JSON keys must be strings
            'next_id': self.next_id,
            'version': '1.0'
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    def load_vocabulary(self, path: Path) -> None:
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.tag_to_id = vocab_data['tag_to_id']
        # Convert string keys back to integers for id_to_tag
        self.id_to_tag = {int(k): v for k, v in vocab_data['id_to_tag'].items()}
        self.next_id = vocab_data['next_id']

    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about the vocabulary."""
        return {
            'total_tags': len(self.tag_to_id),
            'max_id': max(self.id_to_tag.keys()) if self.id_to_tag else -1,
            'next_id': self.next_id
        }

    def contains_tag(self, tag: str) -> bool:
        """Check if a tag exists in the vocabulary."""
        return tag in self.tag_to_id

    def contains_id(self, tag_id: int) -> bool:
        """Check if a token ID exists in the vocabulary."""
        return tag_id in self.id_to_tag

    def get_all_tags(self) -> List[str]:
        """Get list of all tags in vocabulary, sorted by ID."""
        return [self.id_to_tag[i] for i in sorted(self.id_to_tag.keys())]

    def get_all_ids(self) -> List[int]:
        """Get list of all token IDs in vocabulary."""
        return sorted(self.id_to_tag.keys())

    def merge_vocabulary(self, other: 'TagVocabulary') -> None:
        """Merge another vocabulary into this one. New tags get new IDs."""
        for tag in other.get_all_tags():
            self.add_tag(tag)

    def create_vocab_hash(self) -> str:
        """Create a hash of the vocabulary for version checking."""
        # Sort tags by ID to ensure consistent ordering
        sorted_tags = self.get_all_tags()
        vocab_string = '|'.join(sorted_tags)
        return hashlib.sha256(vocab_string.encode('utf-8')).hexdigest()

    def get_vocab_metadata(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary metadata.

        Returns:
            Dictionary with vocab_size, vocab_version, vocab_hash
        """
        return {
            'vocab_size': len(self.tag_to_id),
            'vocab_version': '1.0',  # Version of vocabulary format
            'vocab_hash': self.create_vocab_hash()
        }

    def dump_vocab(self) -> List[Dict[str, Any]]:
        """Export full vocabulary as structured data.

        Returns:
            List of dictionaries with token_id, label for each vocabulary entry
        """
        return [
            {'token_id': token_id, 'label': tag}
            for token_id, tag in sorted(self.id_to_tag.items())
        ]


def build_vocabulary_from_interrogators(interrogator_dict: Dict[str, any], vocab_path: Path) -> TagVocabulary:
    """
    Build a unified vocabulary from all available interrogators.

    Args:
        interrogator_dict: Dictionary of interrogator instances
        vocab_path: Path where to save the vocabulary

    Returns:
        TagVocabulary instance with all tags from all models
    """
    vocab = TagVocabulary()

    # Collect tags from all interrogators
    for name, interrogator in interrogator_dict.items():
        try:
            # Load the interrogator to access its tags
            if not hasattr(interrogator, 'tags') or interrogator.tags is None:
                interrogator.load()

            if hasattr(interrogator, 'tags') and interrogator.tags is not None:
                # Add tags from the model's tag list
                if 'name' in interrogator.tags.columns:
                    tags = interrogator.tags['name'].tolist()
                    vocab.add_tags_from_list([tag for tag in tags if isinstance(tag, str)])

            # Unload to save memory
            interrogator.unload()

        except Exception as e:
            print(f"Warning: Could not load tags from {name}: {e}")
            continue

    # Save the vocabulary
    vocab.save_vocabulary(vocab_path)
    return vocab