#!/usr/bin/env python3
"""
Test script to verify vector generation and Qdrant insertion in isolation
"""
import os
import sys
import tempfile
import uuid
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import base64
import json

# Import the vector generation and Qdrant functions directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_generator import generate_qdrant_sparse_vector, initialize_qdrant_client

def load_env():
    """Load environment variables."""
    # First try .env in the current directory
    if os.path.exists(".env"):
        load_dotenv(".env")
    # Then try .env in the parent directory (if running from scripts/)
    elif os.path.exists("../.env"):
        load_dotenv("../.env")
    
    return {
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_port": int(os.getenv("QDRANT_PORT")),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION"),
        "example_image": os.getenv("EXAMPLE_IMAGE_PATH")
    }

def test_vector_generation():
    """Test vector generation and Qdrant insertion."""
    env = load_env()
    
    # Create a test result object
    test_result = {
        "file": env["example_image"] or "test/image.jpg",
        "gcs_uri": "gs://test-bucket/test/image.jpg",
        "timestamp": "2023-01-01T12:00:00Z",
        "tags": {"test_tag": 0.9, "another_tag": 0.8}
    }
    
    # Generate vector
    print("Generating vector from test data...")
    vector_data = generate_qdrant_sparse_vector(test_result)
    
    # Print vector format
    print("\nVector format:")
    print(f"Vector length: {len(vector_data['vector'])}")
    if "sparse_vectors" in vector_data:
        print("Sparse vectors included:")
        for name, sparse in vector_data["sparse_vectors"].items():
            print(f"- {name}: {len(sparse['indices'])} non-zero values")
    
    # Connect to Qdrant
    print("\nConnecting to Qdrant...")
    qdrant = initialize_qdrant_client()
    
    # Generate a point ID (same way as in main.py)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, test_result["file"]))
    print(f"Generated point ID: {point_id}")
    
    # Insert into Qdrant
    print("\nInserting into Qdrant...")
    try:
        collection_name = env["qdrant_collection"]
        
        # Insert with the same format as main.py
        qdrant.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": vector_data["vector"],
                "sparse_vectors": vector_data.get("sparse_vectors", {}),
                "payload": vector_data["payload"]
            }]
        )
        
        print("Vector successfully inserted!")
        
        # Verify insertion
        print("\nVerifying insertion...")
        result = qdrant.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        
        if result:
            print("Retrieval successful!")
            print(f"Found point with ID: {result[0].id}")
            if hasattr(result[0], 'payload') and result[0].payload:
                if 'file' in result[0].payload:
                    print(f"File: {result[0].payload['file']}")
                if 'tags' in result[0].payload:
                    print(f"Tags count: {len(result[0].payload['tags'])}")
        else:
            print("Retrieval failed - point not found.")
            
    except Exception as e:
        print(f"Error inserting vector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_generation()