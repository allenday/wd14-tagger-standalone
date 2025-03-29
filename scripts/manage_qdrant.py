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

def create_collection(client, collection_name, overwrite=False):
    """Create a new Qdrant collection."""
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
        # Create collection with sparse vector configuration suitable for tags
        # For sparse vectors, we need a dummy dense vector config and the sparse vector config
        # The max dimensionality can be up to 100K, but only non-zero values are stored
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
        
        # Get a sample point if available
        points = client.scroll(
            collection_name=collection_name,
            limit=1
        )[0]
        
        logger.info(f"Collection '{collection_name}' information:")
        logger.info(f"Points: {info.vectors_count}")
        logger.info(f"Created: {info.creation_time}")
        logger.info(f"Vector configuration: {info.config.params}")
        
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
            logger.info("Collection is empty")
        
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
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
        choices=["create", "delete", "list", "info"],
        help="""Command to execute:
  create: Create a new collection
  delete: Delete an existing collection
  list: List all collections
  info: Get detailed information about a collection"""
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
    
    args = parser.parse_args()
    
    # Load environment variables
    env_vars = load_env()
    
    # Use specified collection or default from env
    collection_name = args.collection or env_vars["collection"]
    
    # Initialize Qdrant client
    client = initialize_qdrant_client(env_vars)
    
    # Execute the requested command
    if args.command == "create":
        create_collection(client, collection_name, args.overwrite)
    
    elif args.command == "delete":
        delete_collection(client, collection_name)
    
    elif args.command == "list":
        list_collections(client)
    
    elif args.command == "info":
        collection_info(client, collection_name)

if __name__ == "__main__":
    main()