#!/usr/bin/env python3
"""
Test script to verify Qdrant connection and collection setup.
"""
import os
import sys
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    print("Error: qdrant-client is not installed. Please run: pip install qdrant-client")
    sys.exit(1)

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
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    return {
        "url": os.getenv("QDRANT_URL"),
        "port": int(os.getenv("QDRANT_PORT")),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection": os.getenv("QDRANT_COLLECTION")
    }

def test_connection(env_vars):
    """Test the connection to Qdrant."""
    print(f"Connecting to Qdrant at {env_vars['url']}:{env_vars['port']}...")
    
    try:
        client = QdrantClient(
            url=env_vars["url"],
            port=env_vars["port"],
            api_key=env_vars["api_key"],
        )
        
        # Test basic connection
        collections = client.get_collections()
        print(f"Connection successful. Found {len(collections.collections)} collections.")
        
        for collection in collections.collections:
            print(f"- {collection.name}")
        
        return client
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_collection(client, collection_name):
    """Check the status of a specific collection."""
    print(f"\nChecking collection: {collection_name}")
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' does not exist.")
            return False
            
        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' exists.")
        print(f"Vectors count: {collection_info.vectors_count}")
        
        # Get vector configuration
        print("\nVector configuration:")
        if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
            if hasattr(collection_info.config.params, 'size'):
                print(f"Size: {collection_info.config.params.size}")
            if hasattr(collection_info.config.params, 'distance'):
                print(f"Distance: {collection_info.config.params.distance}")
        
        # Print raw configuration for debugging
        print("\nRaw collection config:")
        print(f"{collection_info}")
        
        # Get sparse vector configuration - try different attribute names
        sparse_config_found = False
        
        if hasattr(collection_info.config, 'sparse_vectors_config') and collection_info.config.sparse_vectors_config:
            print("\nSparse vector configuration found (sparse_vectors_config):")
            for name, config in collection_info.config.sparse_vectors_config.items():
                print(f"- {name}")
            sparse_config_found = True
        
        if hasattr(collection_info, 'sparse_vectors') and collection_info.sparse_vectors:
            print("\nSparse vector configuration found (sparse_vectors):")
            for name in collection_info.sparse_vectors:
                print(f"- {name}")
            sparse_config_found = True
            
        if not sparse_config_found:
            print("\nNo sparse vector configuration found.")
        
        # List first few vectors if any exist
        try:
            # Try to list vectors anyway, even if vectors_count is None
            print("\nListing up to 5 vectors:")
            points = client.scroll(
                collection_name=collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )[0]
            
            for i, point in enumerate(points, 1):
                print(f"\nVector {i}:")
                print(f"ID: {point.id}")
                if hasattr(point, 'payload') and point.payload:
                    if 'file' in point.payload:
                        print(f"File: {point.payload['file']}")
                    if 'gcs_uri' in point.payload:
                        print(f"GCS URI: {point.payload['gcs_uri']}")
                    if 'tags' in point.payload:
                        print(f"Tags count: {len(point.payload['tags'])}")
        except Exception as e:
            print(f"Error listing vectors: {e}")
        
        return True
    
    except Exception as e:
        print(f"Error checking collection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_insert_vector(client, collection_name):
    """Test inserting a dummy vector into the collection."""
    print("\nTesting vector insertion...")
    
    try:
        # Create a dummy dense vector
        dummy_vector = [0.1] * 100
        
        # Create a valid UUID for the point
        import uuid
        point_id = str(uuid.uuid4())
        print(f"Using point ID: {point_id}")
        
        # Try different approaches based on Qdrant client version
        print("Using Qdrant client with models.PointStruct...")
        try:
            from qdrant_client.http.models import PointStruct
            
            # Create a point using PointStruct
            point = PointStruct(
                id=point_id,
                vector=dummy_vector,
                payload={
                    "file": "test/image.jpg",
                    "gcs_uri": "gs://test-bucket/test/image.jpg",
                    "tags": {"test": 0.9}
                }
            )
            
            # Insert the point
            response = client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            print("Insertion successful using PointStruct.")
        
        except Exception as e:
            print(f"Error with PointStruct: {e}")
            print("Trying direct dictionary approach...")
            
            # Create a simple point without sparse vectors
            point = {
                "id": point_id,
                "vector": dummy_vector,
                "payload": {
                    "file": "test/image.jpg",
                    "gcs_uri": "gs://test-bucket/test/image.jpg",
                    "tags": {"test": 0.9}
                }
            }
            
            # Insert the point
            response = client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            print("Insertion successful using dict approach.")
        
        print("Insertion successful.")
        
        # Verify it was inserted
        print(f"Retrieving point with ID: {point_id}")
        result = client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        
        if result:
            print("Retrieval successful.")
            print(f"Found point with ID: {result[0].id}")
        else:
            print("Retrieval failed - point not found.")
        
        return True
    
    except Exception as e:
        print(f"Error inserting test vector: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Load environment variables
    env_vars = load_env()
    
    # Test connection
    client = test_connection(env_vars)
    if not client:
        print("Aborting tests due to connection failure.")
        sys.exit(1)
    
    # Check collection
    collection_name = env_vars["collection"]
    if not check_collection(client, collection_name):
        print("Collection check failed.")
    
    # Test vector insertion
    if not test_insert_vector(client, collection_name):
        print("Vector insertion test failed.")
    
if __name__ == "__main__":
    main()