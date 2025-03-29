import os
import json
import logging
from pathlib import Path
import tempfile
from enum import Enum
import requests

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Qdrant client
qdrant_client = None

class VectorFormat(Enum):
    DENSE_VECTOR = "dense-vector"
    QDRANT_DENSE = "qdrant-dense-vector"
    QDRANT_SPARSE = "qdrant-sparse-vector"

# Valid categories from the original script
VALID_CATEGORIES = {'year', 'character', 'artist', 'meta', 'copyright', 'general', 'rating'}

def initialize_qdrant_client():
    """Initialize and cache the Qdrant client."""
    global qdrant_client
    
    if qdrant_client is None:
        # Get configuration from environment variables
        qdrant_url = os.environ.get("QDRANT_URL", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", None)
        
        logger.info(f"Initializing Qdrant client at {qdrant_url}:{qdrant_port}")
        
        # Initialize client
        qdrant_client = QdrantClient(
            url=qdrant_url,
            port=qdrant_port,
            api_key=qdrant_api_key,
        )
        
        # Create collection if it doesn't exist
        collection_name = os.environ.get("QDRANT_COLLECTION", "image-vectors")
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating collection {collection_name}")
                # Create collection with sparse vector configuration
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=100,  # Dummy size for the dense vector
                        distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "camie": models.SparseVectorParams()
                    }
                )
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {str(e)}")
            # Continue anyway - might be permissions issue and collection exists
    
    return qdrant_client

def download_tag_mapping():
    """Download or load the tag mapping file."""
    # Default path from the original script
    default_mapping_path = str(Path.home() / '.cache/huggingface/hub/models--Camais03--camie-tagger/blobs/c7dc4e38696a812e593916e3f2e51b92f687f8ea')
    
    # Check if file exists
    if Path(default_mapping_path).exists():
        logger.info(f"Using existing tag mapping file at {default_mapping_path}")
        return default_mapping_path
    
    # Otherwise, download it
    # First create the directory
    cache_dir = Path.home() / '.cache/huggingface/hub/models--Camais03--camie-tagger/blobs'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct link to the file from HuggingFace
    mapping_url = "https://huggingface.co/Camais03/camie-tagger/resolve/main/tag_mapping_v1.json"
    
    try:
        logger.info(f"Downloading tag mapping from {mapping_url}")
        response = requests.get(mapping_url)
        response.raise_for_status()
        
        # Save to file
        with open(default_mapping_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Tag mapping downloaded to {default_mapping_path}")
        return default_mapping_path
    except Exception as e:
        logger.error(f"Error downloading tag mapping: {str(e)}")
        
        # Create a fallback mapping
        logger.warning("Creating fallback tag mapping...")
        tag_mapping = {
            'idx_to_tag': {str(i): f'tag_{i}' for i in range(100)},
            'tag_to_category': {f'tag_{i}': 'general' for i in range(100)}
        }
        
        # Write to the expected path
        with open(default_mapping_path, 'w') as f:
            json.dump(tag_mapping, f)
        
        logger.info(f"Created fallback mapping at {default_mapping_path}")
        return default_mapping_path

def load_reference_mapping(mapping_path):
    """Load the reference mapping from a file."""
    with open(mapping_path) as f:
        return json.load(f)

def create_compact_mapping(ref_data, categories, is_exclude=True):
    """Create a compact mapping of tags to indices."""
    valid_tags = []
    idx_to_tag = ref_data['idx_to_tag']
    tag_to_category = ref_data['tag_to_category']

    for idx in range(len(idx_to_tag)):
        tag = idx_to_tag[str(idx)]
        category = tag_to_category[tag]
        # If excluding, keep tags NOT in categories
        # If including, keep tags IN categories
        if (is_exclude and category not in categories) or (not is_exclude and category in categories):
            valid_tags.append(tag)

    tag_to_compact_idx = {tag: idx for idx, tag in enumerate(valid_tags)}
    return tag_to_compact_idx, len(valid_tags)

def convert_to_dense_array(input_tags, tag_to_compact_idx, array_size):
    """Convert tags to a dense array."""
    result = [0.0] * array_size
    for tag, confidence in input_tags.items():
        if tag in tag_to_compact_idx:
            result[tag_to_compact_idx[tag]] = confidence
    return result

def format_output(dense_array, output_format, original_data):
    """Format the output according to the specified format."""
    if output_format == VectorFormat.DENSE_VECTOR:
        return {"vector": dense_array}

    elif output_format == VectorFormat.QDRANT_DENSE:
        return {
            "vector": dense_array,
            "payload": original_data  # Include complete original data
        }

    elif output_format == VectorFormat.QDRANT_SPARSE:
        # Convert dense to sparse format
        non_zero = [(idx, val) for idx, val in enumerate(dense_array) if val > 0]
        
        # Create a dummy dense vector required by Qdrant
        dummy_vector = [0.0] * 100
        
        if non_zero:
            indices, values = zip(*non_zero)
            return {
                "vector": dummy_vector,  # Required dummy dense vector
                "sparse_vectors": {
                    "camie": {  # Match the name defined in the collection
                        "indices": list(indices),
                        "values": list(values)
                    }
                },
                "payload": original_data  # Include complete original data
            }
        else:
            # Handle empty vector case
            return {
                "vector": dummy_vector,  # Required dummy dense vector
                "sparse_vectors": {
                    "camie": {  # Match the name defined in the collection
                        "indices": [],
                        "values": []
                    }
                },
                "payload": original_data
            }

def generate_qdrant_sparse_vector(input_data, exclude_categories=None):
    """Generate a Qdrant sparse vector from input data."""
    if exclude_categories is None:
        # Default to exclude meta categories (can be customized)
        exclude_categories = ['meta']
    
    # Download or load tag mapping
    mapping_path = download_tag_mapping()
    
    # Load reference data
    ref_data = load_reference_mapping(mapping_path)
    
    # Create compact mapping
    tag_to_compact_idx, array_size = create_compact_mapping(
        ref_data, 
        set(exclude_categories), 
        is_exclude=True
    )
    
    # Convert to dense array
    input_tags = input_data.get('tags', {})
    dense_array = convert_to_dense_array(input_tags, tag_to_compact_idx, array_size)
    
    # Format as Qdrant sparse vector
    return format_output(dense_array, VectorFormat.QDRANT_SPARSE, input_data)