"""
Test enhanced API features including batch processing, logits, streaming, and device selection.
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import patch

from tagger import (
    WD14Tagger,
    tag_images,
    get_model_info,
    get_vocab,
    tag_image as standalone_tag_image
)


@pytest.fixture
def test_image_path():
    """Return path to test image."""
    # Use the test.png from root directory if it exists
    test_path = Path("./test.png")
    if test_path.exists():
        return str(test_path)

    # Fallback to creating a simple test image
    img = Image.new('RGB', (64, 64), color='red')
    test_path = Path("/tmp/test_api.png")
    img.save(test_path)
    return str(test_path)


@pytest.fixture
def test_image_array():
    """Return test image as numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def tagger():
    """Return WD14Tagger instance."""
    return WD14Tagger('wd14-convnextv2.v1', quiet=True)


class TestEnhancedAPI:
    """Test enhanced API features."""

    def test_device_selection(self):
        """Test device selection options."""
        # CPU device
        cpu_tagger = WD14Tagger('wd14-convnextv2.v1', device='cpu', quiet=True)
        assert 'CPUExecutionProvider' in cpu_tagger.interrogator.providers

        # MPS device (should work on macOS)
        mps_tagger = WD14Tagger('wd14-convnextv2.v1', device='mps', quiet=True)
        providers = mps_tagger.interrogator.providers
        assert 'CoreMLExecutionProvider' in providers or 'CPUExecutionProvider' in providers

        # Invalid device
        with pytest.raises(ValueError, match="Unsupported device"):
            WD14Tagger('wd14-convnextv2.v1', device='invalid')

    def test_model_info_enhanced(self, tagger):
        """Test enhanced get_model_info functionality."""
        info = tagger.get_model_info()

        # Check required keys
        required_keys = [
            'model', 'version', 'display_name', 'is_loaded', 'providers',
            'vocab_size', 'vocab_hash', 'id_order', 'deterministic'
        ]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        # Check types and values
        assert isinstance(info['model'], str)
        assert isinstance(info['vocab_size'], int)
        assert info['vocab_size'] > 0
        assert isinstance(info['vocab_hash'], str)
        assert len(info['vocab_hash']) == 64  # SHA256 hash
        assert isinstance(info['id_order'], list)
        assert len(info['id_order']) == info['vocab_size']
        assert info['id_order'] == sorted(info['id_order'])  # Should be sorted

    def test_get_vocab(self, tagger):
        """Test get_vocab functionality."""
        vocab = tagger.get_vocab()

        assert isinstance(vocab, dict)
        assert len(vocab) > 0

        # Check that all keys are integers and all values are strings
        for token_id, label in vocab.items():
            assert isinstance(token_id, int)
            assert isinstance(label, str)
            assert token_id >= 0

        # Check that token IDs are sequential
        token_ids = sorted(vocab.keys())
        assert token_ids == list(range(len(token_ids)))

    def test_numpy_array_input(self, tagger, test_image_array):
        """Test numpy array input support."""
        tokens = tagger.tag_image(test_image_array, threshold=0.1)

        assert isinstance(tokens, list)
        for token in tokens:
            assert 'token_id' in token
            assert 'label' in token
            assert 'score' in token
            assert isinstance(token['token_id'], int)
            assert isinstance(token['label'], str)
            assert isinstance(token['score'], float)

    def test_pil_image_input(self, tagger):
        """Test PIL Image input support."""
        pil_image = Image.new('RGB', (64, 64), color='blue')
        tokens = tagger.tag_image(pil_image, threshold=0.1)

        assert isinstance(tokens, list)
        # Should work without errors

    def test_logits_output(self, tagger, test_image_path):
        """Test return_logits functionality."""
        # Single image
        tokens, logits, id_order = tagger.tag_image(
            test_image_path,
            threshold=0.1,
            return_logits=True
        )

        assert isinstance(tokens, list)
        assert isinstance(logits, np.ndarray)
        assert isinstance(id_order, list)
        assert len(logits) == len(id_order)
        assert len(id_order) == tagger.get_model_info()['vocab_size']

    def test_batch_processing(self, tagger, test_image_path):
        """Test batch processing functionality."""
        image_paths = [test_image_path, test_image_path]

        # Without logits
        results = tagger.tag_images(image_paths, threshold=0.1)
        assert len(results) == 2
        assert all(isinstance(tokens, list) for tokens in results)

        # With logits
        results, logits_batch, id_order = tagger.tag_images(
            image_paths,
            threshold=0.1,
            return_logits=True
        )
        assert len(results) == 2
        assert logits_batch.shape[0] == 2  # Batch dimension
        assert logits_batch.shape[1] == len(id_order)  # Vocab dimension

    def test_streaming_api(self, tagger, test_image_path):
        """Test streaming iterator functionality."""
        image_paths = [test_image_path] * 3

        results = []
        for tokens in tagger.iter_tag_images(iter(image_paths), threshold=0.1):
            results.append(tokens)
            assert isinstance(tokens, list)

        assert len(results) == 3

    def test_convenience_functions(self, test_image_path):
        """Test standalone convenience functions."""
        # get_model_info
        info = get_model_info('wd14-convnextv2.v1')
        assert 'vocab_size' in info

        # get_vocab
        vocab = get_vocab('wd14-convnextv2.v1')
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

        # tag_images
        results = tag_images([test_image_path], threshold=0.1)
        assert len(results) == 1
        assert isinstance(results[0], list)

        # tag_images with logits
        results, logits, id_order = tag_images(
            [test_image_path],
            threshold=0.1,
            return_logits=True
        )
        assert isinstance(logits, np.ndarray)
        assert isinstance(id_order, list)

    def test_legacy_compatibility(self, tagger, test_image_path):
        """Test that legacy tag_images_batch still works."""
        results = tagger.tag_images_batch([test_image_path], threshold=0.1)
        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_schema_compliance(self, tagger, test_image_path):
        """Test that all outputs comply with documented schemas."""
        # Token schema
        tokens = tagger.tag_image(test_image_path, threshold=0.1)
        for token in tokens:
            assert set(token.keys()) == {'token_id', 'label', 'score'}
            assert isinstance(token['token_id'], int)
            assert isinstance(token['label'], str)
            assert isinstance(token['score'], float)
            assert 0.0 <= token['score'] <= 1.0

        # Model info schema
        info = tagger.get_model_info()
        expected_keys = {
            'model', 'version', 'display_name', 'is_loaded', 'providers',
            'vocab_size', 'vocab_hash', 'id_order', 'deterministic'
        }
        assert set(info.keys()) == expected_keys

        # Vocab schema
        vocab = tagger.get_vocab()
        for token_id, label in vocab.items():
            assert isinstance(token_id, int)
            assert isinstance(label, str)

    def test_vocabulary_stability(self, tagger):
        """Test that vocabulary hash is stable across calls."""
        info1 = tagger.get_model_info()
        info2 = tagger.get_model_info()

        assert info1['vocab_hash'] == info2['vocab_hash']
        assert info1['vocab_size'] == info2['vocab_size']
        assert info1['id_order'] == info2['id_order']

    def test_error_handling(self, tagger):
        """Test error handling for invalid inputs."""
        # Invalid image path
        with pytest.raises((FileNotFoundError, OSError)):
            tagger.tag_image("/nonexistent/path.jpg")

        # Invalid numpy array shape
        invalid_array = np.random.randint(0, 255, (10,), dtype=np.uint8)
        with pytest.raises(ValueError):
            tagger.tag_image(invalid_array)

    def test_context_manager(self, test_image_path):
        """Test context manager functionality."""
        with WD14Tagger('wd14-convnextv2.v1', quiet=True) as tagger:
            assert tagger.is_loaded
            tokens = tagger.tag_image(test_image_path, threshold=0.1)
            assert len(tokens) >= 0

        # Model should be unloaded after context exit
        assert not tagger.is_loaded