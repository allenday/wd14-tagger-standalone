# API Schema Guarantees

This document defines the stable API schemas and guarantees for the WD14 Tagger library.

## Token Objects

All tagging functions return tokens as structured objects with these exact keys:

```python
{
    "token_id": int,    # Stable integer ID for the tag (0-based, sequential)
    "label": str,       # Human-readable tag string
    "score": float      # Confidence score (0.0-1.0)
}
```

**Guarantees:**
- `token_id` values are stable within a model and vocabulary version
- `token_id` values are sequential starting from 0
- `label` strings match the original model tags
- `score` values are normalized confidence scores from the model

## Model Info Schema

`get_model_info()` returns:

```python
{
    "model": str,           # Model identifier (e.g., "wd14-convnextv2.v1")
    "version": str,         # Vocabulary format version
    "display_name": str,    # Human-readable model name
    "is_loaded": bool,      # Whether model is currently loaded in memory
    "providers": List[str], # ONNX execution providers
    "vocab_size": int,      # Total number of tags in vocabulary
    "vocab_hash": str,      # SHA256 hash of vocabulary for version checking
    "id_order": List[int],  # Ordered list of all token IDs
    "deterministic": bool   # Whether deterministic mode is enabled
}
```

## Vocabulary Schema

`get_vocab()` returns:

```python
Dict[int, str]  # Mapping from token_id to label, sorted by token_id
```

`dump_vocab()` returns:

```python
[
    {"token_id": int, "label": str},
    ...
]  # Sorted by token_id
```

## Logits Schema

When `return_logits=True`, functions return:

```python
(
    tokens: List[Dict[str, Any]],  # Token objects as above
    logits: np.ndarray,            # Raw model outputs in id_order
    id_order: List[int]            # Token ID ordering for logits array
)
```

**Guarantees:**
- `logits` array length equals `len(id_order)`
- `logits[i]` corresponds to token ID `id_order[i]`
- `id_order` is sorted and matches vocabulary ordering
- `logits` values are raw model outputs (not post-processed)

## Batch Operations

Batch functions maintain consistent schemas:

```python
# tag_images() without logits
List[List[Dict[str, Any]]]  # List of token lists

# tag_images() with logits
(
    List[List[Dict[str, Any]]],  # List of token lists
    np.ndarray,                  # Shape: (batch_size, vocab_size)
    List[int]                    # Token ID ordering (same for all images)
)
```

## Input Formats

Supported image input types:
- `str | Path`: File path to image
- `PIL.Image.Image`: PIL Image object
- `np.ndarray`: Numpy array (uint8, shape: HxW or HxWxC)

## Device Selection

Supported device values:
- `None`: Auto-select optimal provider
- `"cpu"`: Force CPU execution
- `"cuda"`: Prefer CUDA if available
- `"mps"`: Prefer CoreML/Metal if available

## Stability Guarantees

1. **Token IDs**: Stable within same model and vocabulary version
2. **Schema Keys**: Will not change in compatible versions
3. **Ordering**: `id_order` provides consistent logits indexing
4. **Hash**: `vocab_hash` changes only when vocabulary changes
5. **Backward Compatibility**: New optional parameters will have sensible defaults

## Version Compatibility

- Major version changes may break schema compatibility
- Minor version changes preserve schema compatibility
- Patch versions only fix bugs, maintain full compatibility