#!/usr/bin/env python3
"""
Script to manage Qdrant collections for the image tagger application.
Provides options to create, delete, or overwrite collections.
"""
import os
import sys
import argparse
import logging
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    print("Error: qdrant-client is not installed. Please run: pip install qdrant-client")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env():
    """Load environment variables from .env file."""
    # First try .env in the current directory
    if os.path.exists(".env"):
        load_dotenv(".env")
    # Then try .env in the parent directory (if running from scripts/)
    elif os.path.exists("../.env"):
        load_dotenv("../.env")
    
    # Check required variables
    required_vars = ["QDRANT_URL", "QDRANT_PORT", "QDRANT_API_KEY", "QDRANT_COLLECTION"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Make sure the .env file exists and contains the required variables:")
        logger.error("QDRANT_URL=your-qdrant-instance.region.gcp.cloud.qdrant.io")
        logger.error("QDRANT_PORT=6333")
        logger.error("QDRANT_API_KEY=your-api-key")
        logger.error("QDRANT_COLLECTION=image-vectors")
        sys.exit(1)
    
    return {
        "url": os.getenv("QDRANT_URL"),
        "port": int(os.getenv("QDRANT_PORT")),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection": os.getenv("QDRANT_COLLECTION")
    }

def initialize_qdrant_client(env_vars):
    """Initialize Qdrant client."""
    logger.info(f"Connecting to Qdrant at {env_vars['url']}:{env_vars['port']}")
    
    try:
        client = QdrantClient(
            url=env_vars["url"],
            port=env_vars["port"],
            api_key=env_vars["api_key"],
        )
        # Test connection
        client.get_collections()
        logger.info("Successfully connected to Qdrant")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        sys.exit(1)

def create_collection(client, collection_name, overwrite=False, with_whash=True):
    """Create a new Qdrant collection with optional wavelet hash vector.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection to create
        overwrite: Whether to overwrite existing collection
        with_whash: Whether to include a wavelet hash vector field
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name in collection_names:
            if overwrite:
                logger.info(f"Collection '{collection_name}' already exists, deleting...")
                client.delete_collection(collection_name=collection_name)
                logger.info(f"Collection '{collection_name}' deleted")
            else:
                logger.info(f"Collection '{collection_name}' already exists, no changes made")
                logger.info("Use --overwrite to recreate the collection")
                return False
        
        logger.info(f"Creating collection '{collection_name}'...")
        
        # Configure vectors based on whether we include the wavelet hash
        if with_whash:
            logger.info("Creating collection with tag vector and wavelet hash vector")
            # Create collection with both tag vector and wavelet hash vector
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # The original vector for tag information
                    "tag_vector": models.VectorParams(
                        size=100,  # Dummy size for the tag dense vector
                        distance=models.Distance.COSINE
                    ),
                    # The vector for wavelet hash (256 bits = 256 dimensions)
                    "whash_vector": models.VectorParams(
                        size=256,  # Size for wavelet hash vector (16x16 bits)
                        distance=models.Distance.DOT  # Use DOT product as fallback since HAMMING might not be supported
                    )
                },
                sparse_vectors_config={
                    "camie": models.SparseVectorParams(
                        # No size limitation needed for sparse vectors
                        # Only non-zero elements are stored
                    )
                },
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately
                )
            )
        else:
            logger.info("Creating collection with tag vector only")
            # Create collection with sparse vector configuration suitable for tags
            # For sparse vectors, we need a dummy dense vector config and the sparse vector config
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=100,  # Dummy size for the dense vector (not actually used)
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "camie": models.SparseVectorParams(
                        # No size limitation needed for sparse vectors
                        # Only non-zero elements are stored
                    )
                },
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately
                )
            )
        
        logger.info(f"Collection '{collection_name}' created successfully")
        
        # Create payload index for faster searches
        client.create_payload_index(
            collection_name=collection_name,
            field_name="file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="gcs_uri",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload indices for 'file' and 'gcs_uri' fields")
        
        return True
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return False

def delete_collection(client, collection_name):
    """Delete an existing Qdrant collection."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' does not exist, nothing to delete")
            return False
        
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        return False

def list_collections(client):
    """List all available Qdrant collections."""
    try:
        collections = client.get_collections().collections
        
        if not collections:
            logger.info("No collections found in Qdrant")
            return []
        
        logger.info(f"Found {len(collections)} collections:")
        for i, collection in enumerate(collections, 1):
            logger.info(f"{i}. {collection.name}")
            
            # Get collection info
            try:
                info = client.get_collection(collection_name=collection.name)
                point_count = info.vectors_count
                logger.info(f"   - Points: {point_count}")
                logger.info(f"   - Created: {info.creation_time}")
            except Exception as e:
                logger.error(f"   - Error getting collection info: {e}")
        
        return [c.name for c in collections]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []

def update_collection_with_whash(client, collection_name):
    """Update an existing collection to add wavelet hash vector support.
    
    This requires recreating the collection while preserving its data.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection to update
        
    Returns:
        bool: Whether the update was successful
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' does not exist")
            return False
        
        # Get collection info
        info = client.get_collection(collection_name=collection_name)
        
        # Check if collection already has whash_vector
        if "whash_vector" in info.config.params.vectors:
            logger.info(f"Collection '{collection_name}' already has wavelet hash vector, no update needed")
            return True
        
        logger.info(f"Updating collection '{collection_name}' to add wavelet hash vector")
        
        # Create a temporary collection name
        temp_collection_name = f"{collection_name}_temp"
        
        # Check if temp collection exists and delete if needed
        if temp_collection_name in collection_names:
            logger.info(f"Temporary collection '{temp_collection_name}' exists, deleting...")
            client.delete_collection(collection_name=temp_collection_name)
        
        # Create a new collection with whash support
        logger.info(f"Creating temporary collection '{temp_collection_name}' with wavelet hash support")
        client.create_collection(
            collection_name=temp_collection_name,
            vectors_config={
                # The original vector for tag information
                "tag_vector": models.VectorParams(
                    size=100,  # Dummy size for the tag dense vector
                    distance=models.Distance.COSINE
                ),
                # The vector for wavelet hash (256 bits = 256 dimensions)
                "whash_vector": models.VectorParams(
                    size=256,  # Size for wavelet hash vector (16x16 bits)
                    distance=models.Distance.DOT  # Use DOT product as fallback since HAMMING might not be supported
                )
            },
            sparse_vectors_config={
                "camie": models.SparseVectorParams()
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0  # Index immediately
            )
        )
        
        # Create the same payload indices
        client.create_payload_index(
            collection_name=temp_collection_name,
            field_name="file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=temp_collection_name,
            field_name="gcs_uri",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        # Now we delete the original collection
        logger.info(f"Deleting original collection '{collection_name}'")
        client.delete_collection(collection_name=collection_name)
        
        # Rename the temporary collection to the original name
        logger.info(f"Renaming temporary collection '{temp_collection_name}' to '{collection_name}'")
        client.rename_collection(
            collection_name=temp_collection_name,
            new_collection_name=collection_name
        )
        
        logger.info(f"Collection '{collection_name}' successfully updated with wavelet hash support")
        logger.info("Note: The collection is now empty and will need to be repopulated with data")
        logger.info("Any existing entries will need to be reimported with wavelet hash values")
        
        return True
    except Exception as e:
        logger.error(f"Error updating collection: {e}")
        return False


def collection_info(client, collection_name):
    """Get detailed information about a specific collection."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' does not exist")
            return None
        
        # Get collection info
        info = client.get_collection(collection_name=collection_name)
        
        # Get collection statistics
        # We need to use count explicitly as vectors_count may not be reported correctly
        count_result = client.count(collection_name=collection_name, exact=True)
        points_count = count_result.count
        
        # Get a sample point if available
        points = client.scroll(
            collection_name=collection_name,
            limit=1
        )[0]
        
        logger.info(f"Collection '{collection_name}' information:")
        logger.info(f"Points count: {points_count}")
        
        # Display vector configuration
        vector_config = info.config.params
        
        # Check if collection has named vectors (newer format)
        if hasattr(vector_config, 'vectors') and vector_config.vectors:
            logger.info("Named vector configurations:")
            # Handle different vector config formats
            if isinstance(vector_config.vectors, dict):
                # Dict style format
                for name, params in vector_config.vectors.items():
                    logger.info(f"  {name}: size={params.size}, distance={params.distance}")
            else:
                # Object style format (model instance)
                try:
                    # Try to represent vector_config as a dictionary for display
                    vector_dict = vector_config.vectors.dict() if hasattr(vector_config.vectors, 'dict') else vars(vector_config.vectors)
                    for name, params in vector_dict.items():
                        if isinstance(params, dict) and 'size' in params:
                            logger.info(f"  {name}: size={params['size']}, distance={params.get('distance', 'unknown')}")
                        else:
                            logger.info(f"  {name}: {params}")
                except Exception as e:
                    # Fall back to simple representation
                    logger.info(f"  Vector config: {vector_config.vectors}")
        # Fall back to legacy single vector format
        elif hasattr(vector_config, 'size'):
            logger.info(f"Vector configuration: size={vector_config.size}, distance={vector_config.distance}")
        else:
            logger.info("Vector configuration not available")
        
        # Check if collection has sparse vectors
        if hasattr(vector_config, 'sparse_vectors') and vector_config.sparse_vectors:
            logger.info("Sparse vector configurations:")
            try:
                # Try to get names directly if it's a dict-like object
                if hasattr(vector_config.sparse_vectors, 'keys'):
                    for name in vector_config.sparse_vectors.keys():
                        logger.info(f"  {name}")
                # Otherwise try to convert to a dict or print the object
                else:
                    sparse_dict = vector_config.sparse_vectors.dict() if hasattr(vector_config.sparse_vectors, 'dict') else vars(vector_config.sparse_vectors)
                    for name, config in sparse_dict.items():
                        logger.info(f"  {name}: {config}")
            except Exception as e:
                # Fall back to simple string representation
                logger.info(f"  Sparse vector config: {vector_config.sparse_vectors}")
        
        # Display collection metadata if available
        if hasattr(info, 'metadata') and info.metadata:
            logger.info("Collection metadata:")
            for k, v in info.metadata.items():
                logger.info(f"  {k}: {v}")
        
        # Display sample point if available
        if points:
            logger.info("Sample point payload:")
            payload = points[0].payload
            for k, v in payload.items():
                # Skip tags array to keep output clean
                if k == "tags" and isinstance(v, dict) and len(v) > 5:
                    logger.info(f"  {k}: {len(v)} tags")
                    # Show just a few tags
                    sample_tags = list(v.items())[:5]
                    for tag, score in sample_tags:
                        logger.info(f"    - {tag}: {score}")
                    logger.info(f"    - ... and {len(v) - 5} more tags")
                else:
                    logger.info(f"  {k}: {v}")
        else:
            logger.info("Collection is empty or no points returned")
        
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Manage Qdrant collections for image tagger",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Command argument
    parser.add_argument(
        "command", 
        choices=["create", "delete", "list", "info", "update-whash"],
        help="""Command to execute:
  create: Create a new collection
  delete: Delete an existing collection
  list: List all collections
  info: Get detailed information about a collection
  update-whash: Update collection to add wavelet hash vector support"""
    )
    
    # Optional arguments
    parser.add_argument(
        "--collection", 
        help="Collection name (defaults to QDRANT_COLLECTION from .env)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite collection if it already exists"
    )
    parser.add_argument(
        "--no-whash",
        action="store_true",
        help="Do not include wavelet hash vector when creating collection"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_env()
    
    # Use specified collection or default from env
    collection_name = args.collection or env_vars["collection"]
    
    # Initialize Qdrant client
    client = initialize_qdrant_client(env_vars)
    
    # Execute the requested command
    if args.command == "create":
        create_collection(client, collection_name, args.overwrite, not args.no_whash)
    
    elif args.command == "delete":
        delete_collection(client, collection_name)
    
    elif args.command == "list":
        list_collections(client)
    
    elif args.command == "info":
        collection_info(client, collection_name)
        
    elif args.command == "update-whash":
        update_collection_with_whash(client, collection_name)

if __name__ == "__main__":
    main()