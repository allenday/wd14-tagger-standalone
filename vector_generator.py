import os
import json
import logging
from pathlib import Path
import tempfile
from enum import Enum
import requests
import base64
from io import BytesIO
import numpy as np

import imagehash
from PIL import Image
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
    QDRANT_COMBINED = "qdrant-combined-vectors"  # Includes both sparse tag vectors and perceptual hash

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
                # Create collection with both sparse vector for tags and a vector for wavelet hash
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        # The original vector for tag information
                        "tag_vector": models.VectorParams(
                            size=100,  # Dummy size for the tag dense vector
                            distance=models.Distance.COSINE
                        ),
                        # The new vector for wavelet hash (256 bits = 256 dimensions)
                        "whash_vector": models.VectorParams(
                            size=256,  # Size for wavelet hash vector (16x16 bits)
                            distance=models.Distance.DOT  # Use DOT product as fallback since HAMMING might not be supported
                        )
                    },
                    sparse_vectors_config={
                        "camie": models.SparseVectorParams()
                    }
                )
                logger.info("Created collection with tag vector and wavelet hash vector")
            else:
                # Check if collection needs to be updated with wavelet hash vector
                try:
                    collection_info = qdrant_client.get_collection(collection_name=collection_name)
                    if "whash_vector" not in collection_info.config.params.vectors:
                        logger.info(f"Recreating collection {collection_name} to add wavelet hash vector")
                        # We need to recreate the collection with the new vector configuration
                        # First delete the existing one
                        qdrant_client.delete_collection(collection_name=collection_name)
                        
                        # Then create it with both vectors
                        qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config={
                                # The original vector for tag information
                                "tag_vector": models.VectorParams(
                                    size=100,  # Dummy size for the tag dense vector
                                    distance=models.Distance.COSINE
                                ),
                                # The new vector for wavelet hash (256 bits = 256 dimensions)
                                "whash_vector": models.VectorParams(
                                    size=256,  # Size for wavelet hash vector (16x16 bits)
                                    distance=models.Distance.HAMMING  # Hamming distance is better for binary hashes
                                )
                            },
                            sparse_vectors_config={
                                "camie": models.SparseVectorParams()
                            }
                        )
                        logger.info("Recreated collection with both tag vector and wavelet hash vector")
                except Exception as e:
                    logger.error(f"Error checking collection structure: {str(e)}")
                    # Continue anyway - might be permissions issue
                    
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

def calculate_wavelet_hash(image_data):
    """Calculate a wavelet hash (whash) for an image.
    
    Args:
        image_data: Binary image data
        
    Returns:
        A list of 256 binary values (0 or 1) representing the 16x16 wavelet hash
    """
    try:
        # Load image from binary data
        image = Image.open(BytesIO(image_data))
        
        # Calculate wavelet hash
        hash_obj = imagehash.whash(image, hash_size=16)  # 16x16 = 256 bits
        
        # Convert hash to binary vector (list of 0s and 1s)
        # The hash is stored as a 64-bit hexadecimal number
        hash_binary = []
        
        # Convert each hex digit to 4 binary digits
        for hex_digit in str(hash_obj):
            # Skip any non-hex characters
            if hex_digit not in '0123456789abcdef':
                continue
                
            # Convert hex digit to int
            value = int(hex_digit, 16)
            
            # Convert to 4 binary digits
            for i in range(3, -1, -1):
                bit = (value >> i) & 1
                hash_binary.append(float(bit))  # Use float for consistency with other vectors
                
        # Ensure we have exactly 256 dimensions
        if len(hash_binary) < 256:
            hash_binary.extend([0.0] * (256 - len(hash_binary)))
        elif len(hash_binary) > 256:
            hash_binary = hash_binary[:256]
            
        logger.info(f"Generated wavelet hash with {len(hash_binary)} dimensions")
        return hash_binary
        
    except Exception as e:
        logger.error(f"Error calculating wavelet hash: {str(e)}")
        # Return a zero vector if hash calculation fails
        return [0.0] * 256


def get_image_hash_similarity(image_data1, image_data2):
    """Calculate the similarity between two images based on their wavelet hashes.
    
    Args:
        image_data1: Binary data for the first image
        image_data2: Binary data for the second image
        
    Returns:
        A float between 0 and 1, where 1 means identical images and 0 means completely different.
    """
    try:
        # Open images
        img1 = Image.open(BytesIO(image_data1))
        img2 = Image.open(BytesIO(image_data2))
        
        # Calculate hashes
        hash1 = imagehash.whash(img1, hash_size=16)
        hash2 = imagehash.whash(img2, hash_size=16)
        
        # Calculate Hamming distance (number of different bits)
        # hash1 - hash2 returns the Hamming distance
        distance = hash1 - hash2
        
        # Convert to similarity score (0 to 1)
        # The maximum distance for 256-bit hashes is 256
        similarity = 1.0 - (distance / 256.0)
        
        return similarity
    except Exception as e:
        logger.error(f"Error calculating image similarity: {str(e)}")
        return 0.0


def search_by_image(image_data, limit=10, score_threshold=0.85):
    """Search for similar images in Qdrant using wavelet hash.
    
    Args:
        image_data: Binary data for the query image
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score to include in results
        
    Returns:
        List of similar images with their similarity scores and metadata
    """
    try:
        # Initialize Qdrant client
        qdrant_client = initialize_qdrant_client()
        collection_name = os.environ.get("QDRANT_COLLECTION", "image-vectors")
        
        # Calculate wavelet hash for the query image
        hash_vector = calculate_wavelet_hash(image_data)
        
        # Search in Qdrant using the wavelet hash
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("whash_vector", hash_vector),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        # Process and return results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching by image: {str(e)}")
        return []


def format_output(dense_array, output_format, original_data, image_data=None):
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
    
    elif output_format == VectorFormat.QDRANT_COMBINED:
        # This format includes both the tag vector and the wavelet hash vector
        
        # First, process the tag vector part (same as QDRANT_SPARSE)
        non_zero = [(idx, val) for idx, val in enumerate(dense_array) if val > 0]
        
        # Create a dummy dense vector required by Qdrant
        dummy_vector = [0.0] * 100
        
        # Get the whash vector if we have image data
        whash_vector = calculate_wavelet_hash(image_data) if image_data else [0.0] * 256
        
        # Prepare the response
        if non_zero:
            indices, values = zip(*non_zero)
            return {
                "vectors": {
                    "tag_vector": dummy_vector,  # Required dummy dense vector
                    "whash_vector": whash_vector  # Wavelet hash vector
                },
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
                "vectors": {
                    "tag_vector": dummy_vector,  # Required dummy dense vector
                    "whash_vector": whash_vector  # Wavelet hash vector
                },
                "sparse_vectors": {
                    "camie": {  # Match the name defined in the collection
                        "indices": [],
                        "values": []
                    }
                },
                "payload": original_data
            }

def generate_qdrant_sparse_vector(input_data, exclude_categories=None, image_data=None, include_whash=True):
    """Generate a Qdrant sparse vector from input data.
    
    Args:
        input_data: Dictionary containing tags and other metadata
        exclude_categories: List of categories to exclude from the vector
        image_data: Optional binary image data for calculating wavelet hash
        include_whash: Whether to include the wavelet hash in the output
        
    Returns:
        Dictionary with vector and payload data formatted for Qdrant
    """
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
    
    # Choose output format based on whether we're including the wavelet hash
    output_format = VectorFormat.QDRANT_COMBINED if include_whash and image_data else VectorFormat.QDRANT_SPARSE
    
    # Format the output with the appropriate format
    return format_output(dense_array, output_format, input_data, image_data)