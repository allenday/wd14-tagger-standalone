# WD14 Tagger Enhanced API Documentation

This document describes the enhanced API features available in WD14 Tagger v0.1.7+.

## Overview

The WD14 Tagger provides both CLI and programmatic interfaces for image tagging using WD14 models. The enhanced API includes stable token IDs, batch processing, logits access, streaming support, and comprehensive device control.

## Quick Start

```python
from tagger import WD14Tagger, tag_images, get_model_info

# Basic usage
tagger = WD14Tagger('wd14-convnextv2.v1')
tokens = tagger.tag_image('image.jpg')
print(tokens)
# [{'token_id': 4, 'label': '1girl', 'score': 0.994}, ...]

# Batch processing with logits
tokens_batch, logits, id_order = tag_images(
    ['img1.jpg', 'img2.jpg'],
    return_logits=True
)
```

## Core API Classes

### WD14Tagger

The main class for image tagging operations.

```python
class WD14Tagger:
    def __init__(
        self,
        model_name: str = 'wd14-convnextv2.v1',
        device: Optional[str] = None,
        deterministic: bool = False,
        quiet: bool = True
    )
```

#### Parameters
- `model_name`: Model identifier (see `get_available_models()`)
- `device`: Device selection (`'cpu'`, `'cuda'`, `'mps'`, or `None` for auto)
- `deterministic`: Enable deterministic computation if supported
- `quiet`: Suppress logging output

#### Methods

##### `tag_image()`

Tag a single image with comprehensive options.

```python
def tag_image(
    self,
    image_input: Union[str, Path, Image.Image, np.ndarray],
    threshold: float = 0.35,
    return_logits: bool = False,
    **kwargs
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray, List[int]]]
```

**Parameters:**
- `image_input`: Image file path, PIL Image, or numpy array
- `threshold`: Confidence threshold for tag inclusion
- `return_logits`: Also return raw model logits in vocabulary order
- `**kwargs`: Additional processing options

**Returns:**
- Without logits: `List[Dict[str, Any]]` - Token objects
- With logits: `(tokens, logits_array, id_order)`

**Token Schema:**
```python
{
    "token_id": int,    # Stable integer ID (0-based, sequential)
    "label": str,       # Human-readable tag string
    "score": float      # Confidence score (0.0-1.0)
}
```

##### `tag_images()`

Batch process multiple images efficiently.

```python
def tag_images(
    self,
    image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
    threshold: float = 0.35,
    return_logits: bool = False,
    **kwargs
) -> Union[List[List[Dict[str, Any]]], Tuple[List[List[Dict[str, Any]]], np.ndarray, List[int]]]
```

**Parameters:**
- `image_inputs`: List of images (paths, PIL Images, or numpy arrays)
- `threshold`: Confidence threshold for tag inclusion
- `return_logits`: Also return batched logits with consistent id_order
- `**kwargs`: Additional processing options

**Returns:**
- Without logits: `List[List[Dict[str, Any]]]` - List of token lists
- With logits: `(token_lists, logits_batch, id_order)`
  - `logits_batch`: Shape `(batch_size, vocab_size)`
  - `id_order`: Token ID ordering for logits array

##### `iter_tag_images()`

Stream tag results for large image sets to avoid loading all into memory.

```python
def iter_tag_images(
    self,
    image_inputs: Iterator[Union[str, Path, Image.Image, np.ndarray]],
    threshold: float = 0.35,
    **kwargs
) -> Iterator[List[Dict[str, Any]]]
```

**Use Case:**
```python
# Process large dataset without memory issues
for tokens in tagger.iter_tag_images(image_generator(), threshold=0.3):
    process_tokens(tokens)
```

##### `get_vocab()`

Get vocabulary as a stable mapping.

```python
def get_vocab(self) -> Dict[int, str]
```

**Returns:** Dictionary mapping token IDs to tag strings, sorted by ID.

##### `get_model_info()`

Get comprehensive model information.

```python
def get_model_info(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    "model": str,           # Model identifier
    "version": str,         # Vocabulary format version
    "display_name": str,    # Human-readable model name
    "is_loaded": bool,      # Whether model is loaded in memory
    "providers": List[str], # ONNX execution providers
    "vocab_size": int,      # Total number of tags
    "vocab_hash": str,      # SHA256 hash for version checking
    "id_order": List[int],  # Ordered list of all token IDs
    "deterministic": bool   # Whether deterministic mode enabled
}
```

##### `dump_vocab()`

Export full vocabulary as structured data.

```python
def dump_vocab(self) -> List[Dict[str, Any]]
```

**Returns:**
```python
[
    {"token_id": int, "label": str},
    ...
]  # Sorted by token_id
```

## Convenience Functions

For quick operations without instantiating a class:

```python
# Single image tagging
from tagger import tag_image
tokens = tag_image('image.jpg', model_name='wd14-convnextv2.v1')

# Batch processing
from tagger import tag_images
tokens_batch, logits, id_order = tag_images(
    ['img1.jpg', 'img2.jpg'],
    return_logits=True
)

# Model information
from tagger import get_model_info, get_vocab
info = get_model_info('wd14-convnextv2.v1')
vocab = get_vocab('wd14-convnextv2.v1')
```

## Input Format Support

### Image Inputs

The API accepts multiple input formats:

```python
# File path
tokens = tagger.tag_image('path/to/image.jpg')

# PIL Image
from PIL import Image
pil_img = Image.open('image.jpg')
tokens = tagger.tag_image(pil_img)

# NumPy array
import numpy as np
img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
tokens = tagger.tag_image(img_array)
```

### NumPy Array Requirements

- **Shape:** `(H, W)` for grayscale, `(H, W, 3)` for RGB, `(H, W, 4)` for RGBA
- **Data type:** `uint8` preferred (0-255 range)
- **Auto-conversion:** Float arrays (0.0-1.0) are automatically scaled to uint8

## Device Selection

### Automatic Selection

```python
# Auto-select optimal provider (default)
tagger = WD14Tagger('wd14-convnextv2.v1')
```

### Manual Selection

```python
# Force CPU execution
tagger = WD14Tagger('wd14-convnextv2.v1', device='cpu')

# Prefer CUDA if available
tagger = WD14Tagger('wd14-convnextv2.v1', device='cuda')

# Prefer CoreML/Metal (Apple Silicon)
tagger = WD14Tagger('wd14-convnextv2.v1', device='mps')
```

### Provider Priority

1. **`mps`**: CoreMLExecutionProvider → CPUExecutionProvider
2. **`cuda`**: CUDAExecutionProvider → CPUExecutionProvider
3. **`cpu`**: CPUExecutionProvider only
4. **`None`**: Auto-optimal selection based on system capabilities

## Logits Access

### Single Image

```python
tokens, logits, id_order = tagger.tag_image(
    'image.jpg',
    return_logits=True
)

# logits[i] corresponds to token ID id_order[i]
assert len(logits) == len(id_order)
assert logits.shape == (vocab_size,)
```

### Batch Processing

```python
tokens_batch, logits_batch, id_order = tagger.tag_images(
    ['img1.jpg', 'img2.jpg'],
    return_logits=True
)

# Consistent id_order across all images
assert logits_batch.shape == (2, vocab_size)
assert len(id_order) == vocab_size
```

## Vocabulary Management

### Stability Guarantees

- **Token IDs**: Stable within same model and vocabulary version
- **Ordering**: `id_order` provides consistent logits indexing
- **Versioning**: `vocab_hash` changes only when vocabulary changes
- **Sequential**: Token IDs are 0-based and sequential

### Vocabulary Operations

```python
# Get vocabulary mapping
vocab = tagger.get_vocab()
assert vocab[0] == 'general'  # First token

# Check vocabulary metadata
info = tagger.get_model_info()
print(f"Vocab size: {info['vocab_size']}")
print(f"Vocab hash: {info['vocab_hash']}")

# Export vocabulary
vocab_export = tagger.dump_vocab()
# [{'token_id': 0, 'label': 'general'}, ...]
```

## Error Handling

### Common Exceptions

```python
# Invalid model name
try:
    tagger = WD14Tagger('invalid-model')
except ValueError as e:
    print(f"Model error: {e}")

# Invalid device
try:
    tagger = WD14Tagger('wd14-convnextv2.v1', device='invalid')
except ValueError as e:
    print(f"Device error: {e}")

# Invalid image input
try:
    tokens = tagger.tag_image('/nonexistent/path.jpg')
except (FileNotFoundError, OSError) as e:
    print(f"Image error: {e}")

# Invalid numpy array shape
try:
    invalid_array = np.random.randint(0, 255, (10,), dtype=np.uint8)
    tokens = tagger.tag_image(invalid_array)
except ValueError as e:
    print(f"Array error: {e}")
```

## Context Manager Support

```python
# Automatic model loading/unloading
with WD14Tagger('wd14-convnextv2.v1') as tagger:
    tokens = tagger.tag_image('image.jpg')
    # Model automatically unloaded on exit
```

## Performance Considerations

### Memory Management

```python
# For single images, models auto-load on first use
tagger = WD14Tagger('wd14-convnextv2.v1')
tokens = tagger.tag_image('image.jpg')  # Model loads here

# Explicit loading for better control
tagger.load_model()
# ... process many images ...
tagger.unload_model()  # Free memory
```

### Batch Processing

```python
# Efficient batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
tokens_batch = tagger.tag_images(images)  # More efficient than individual calls

# Streaming for large datasets
def image_generator():
    for path in huge_image_list:
        yield path

for tokens in tagger.iter_tag_images(image_generator()):
    save_results(tokens)  # Process incrementally
```

## Advanced Usage

### Custom Processing Parameters

```python
tokens = tagger.tag_image(
    'image.jpg',
    threshold=0.25,                    # Lower threshold
    additional_tags=['custom_tag'],    # Force include tags
    exclude_tags=['unwanted_tag'],     # Exclude specific tags
    sort_by_alphabetical_order=True,   # Sort by name not confidence
    replace_underscore=True,           # Replace _ with spaces
    escape_tag=True                    # Escape special characters
)
```

### Logits Analysis

```python
tokens, logits, id_order = tagger.tag_image('image.jpg', return_logits=True)

# Find top-K predictions (including below threshold)
top_k_indices = np.argsort(logits)[-10:][::-1]
top_k_tokens = [id_order[i] for i in top_k_indices]
top_k_scores = logits[top_k_indices]

vocab = tagger.get_vocab()
for token_id, score in zip(top_k_tokens, top_k_scores):
    print(f"{vocab[token_id]}: {score:.3f}")
```

### Model Comparison

```python
models = ['wd14-convnextv2.v1', 'wd14-vit.v1', 'wd14-swinv2.v1']

for model_name in models:
    info = get_model_info(model_name)
    print(f"{model_name}: {info['vocab_size']} tokens, hash: {info['vocab_hash'][:8]}...")
```

## Schema Reference

### Token Object
```python
{
    "token_id": int,    # 0-based sequential ID
    "label": str,       # Tag string from model
    "score": float      # Confidence (0.0-1.0)
}
```

### Model Info Object
```python
{
    "model": str,           # Model identifier
    "version": str,         # Vocabulary format version
    "display_name": str,    # Human-readable name
    "is_loaded": bool,      # Memory status
    "providers": List[str], # ONNX providers
    "vocab_size": int,      # Total tags
    "vocab_hash": str,      # SHA256 hash
    "id_order": List[int],  # Token ID ordering
    "deterministic": bool   # Deterministic mode
}
```

### Vocabulary Export
```python
[
    {"token_id": int, "label": str},
    ...
]  # Sorted by token_id
```

## Migration Guide

### From v0.1.6 to v0.1.7

The v0.1.7 API is fully backward compatible. New features are opt-in:

```python
# Old API (still works)
tagger = WD14Tagger('wd14-convnextv2.v1', use_cpu=True)
tokens = tagger.tag_images_batch(['img1.jpg'])

# New API (enhanced)
tagger = WD14Tagger('wd14-convnextv2.v1', device='cpu')
tokens, logits, id_order = tagger.tag_images(['img1.jpg'], return_logits=True)
```

### Key Changes

1. **Constructor**: `use_cpu=True` → `device='cpu'`
2. **Batch Method**: `tag_images_batch()` → `tag_images()` (old method still available)
3. **Logits**: New `return_logits=True` parameter
4. **Input Types**: Now accepts numpy arrays and PIL Images directly
5. **Model Info**: Enhanced with `vocab_hash`, `id_order`, etc.

## Best Practices

1. **Use context managers** for automatic resource management
2. **Batch process** multiple images when possible
3. **Stream large datasets** with `iter_tag_images()`
4. **Cache model info** to avoid repeated vocabulary builds
5. **Specify device** explicitly for consistent performance
6. **Use logits** for advanced analysis and custom thresholding
7. **Monitor vocab_hash** for model version changes