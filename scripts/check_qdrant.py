#!/usr/bin/env python3
"""
Script to check if an image was successfully processed and stored in Qdrant.
"""
import argparse
import os
import sys
import uuid
from pathlib import Path
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
        print("Make sure the .env file exists and contains the required variables:")
        print("QDRANT_URL=your-qdrant-instance.region.gcp.cloud.qdrant.io")
        print("QDRANT_PORT=6333")
        print("QDRANT_API_KEY=your-api-key")
        print("QDRANT_COLLECTION=image-vectors")
        sys.exit(1)
    
    return {
        "url": os.getenv("QDRANT_URL"),
        "port": int(os.getenv("QDRANT_PORT")),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection": os.getenv("QDRANT_COLLECTION")
    }

def check_vector(filepath, custom_env=None, custom_filepath=None):
    """Check if a vector exists for the given filepath.
    
    Args:
        filepath: The local filepath that was processed
        custom_env: Optional custom environment variables
        custom_filepath: Optional custom filepath that was used in the message
    """
    # Load default environment if not provided
    env_vars = custom_env if custom_env else load_env()
    
    # Connect to Qdrant
    client = QdrantClient(
        url=env_vars["url"] if "url" in env_vars else os.getenv("QDRANT_URL"),
        port=int(env_vars["port"] if "port" in env_vars else os.getenv("QDRANT_PORT")),
        api_key=env_vars["api_key"] if "api_key" in env_vars else os.getenv("QDRANT_API_KEY"),
    )
    
    # Use the custom filepath for ID generation if provided
    path_for_id = custom_filepath if custom_filepath else filepath
    
    # Create point ID from filepath (same method used in the Cloud Function)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path_for_id))
    print(f"Checking for vector with ID: {point_id}")
    print(f"Generated from filepath: {path_for_id}")
    if custom_filepath:
        print(f"Original filepath: {filepath}")
    
    collection_name = env_vars["collection"] if "collection" in env_vars else os.getenv("QDRANT_COLLECTION")
    
    # Check collection status
    try:
        # First check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' does not exist.")
            return False
            
        # Check collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' contains {collection_info.vectors_count} vectors.")
        
        # Try to retrieve by ID first
        result = client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        
        # If not found by ID, try to search by filepath in payload
        if not result:
            print("Vector not found by ID. Trying to search by filepath...")
            
            # Create filter for the file path
            file_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="file",
                        match=models.MatchText(text=filepath)
                    )
                ]
            )
            
            search_result = client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
                with_vectors=False,
                scroll_filter=file_filter
            )
            result = search_result[0]  # First element is the points
        
        print(f"Found {len(result)} vectors")
        if not result:
            print("No vectors found for this image. Possible reasons:")
            print("1. The image hasn't been processed yet")
            print("2. The processing failed")
            print("3. The Cloud Function used a different ID generation method")
            
            # Show all points in collection (if any exist and not too many)
            vector_count = collection_info.vectors_count
            if vector_count and vector_count < 10:
                print(f"\nListing all {vector_count} vectors in collection:")
                all_points = client.scroll(
                    collection_name=collection_name,
                    limit=10,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                for i, point in enumerate(all_points, 1):
                    print(f"\nVector {i}:")
                    print(f"ID: {point.id}")
                    if 'file' in point.payload:
                        print(f"Filepath: {point.payload['file']}")
                    if 'gcs_uri' in point.payload:
                        print(f"GCS URI: {point.payload['gcs_uri']}")
            
            return False
        
        # Print metadata
        print("\nVector metadata:")
        print(f"Vector ID: {result[0].id}")
        print(f"Image filepath: {result[0].payload.get('file')}")
        print(f"GCS URI: {result[0].payload.get('gcs_uri')}")
        print(f"Timestamp: {result[0].payload.get('timestamp')}")
        
        # Print tags
        tags = result[0].payload.get("tags", [])
        if tags:
            print(f"\nFound {len(tags)} tags. Top tags:")
            # Tags are already sorted by score in the array
            for i, tag_obj in enumerate(tags[:10], 1):
                print(f"{i}. {tag_obj['name']}: {tag_obj['score']:.4f}")
        else:
            print("\nNo tags found in the vector payload.")
        
        return True
    
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check if an image was processed and stored in Qdrant")
    parser.add_argument("filepath", help="Path to the image file")
    
    args = parser.parse_args()
    
    # Check if file exists
    path = Path(args.filepath)
    if not path.exists() or not path.is_file():
        print(f"Error: File {args.filepath} does not exist or is not a file")
        sys.exit(1)
    
    # Load environment variables
    env_vars = load_env()
    
    # Check vector
    check_vector(str(path), env_vars)
    
if __name__ == "__main__":
    main()