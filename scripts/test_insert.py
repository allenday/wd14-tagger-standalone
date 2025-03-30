#!/usr/bin/env python3
"""
Script to test inserting data with the new collection format.
"""
import os
import sys
import uuid
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    print("Error: qdrant-client is not installed. Please run: pip install qdrant-client")
    sys.exit(1)

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../.env"):
    load_dotenv("../.env")

# Get Qdrant connection details
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")

if not all([QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION]):
    print("Error: Missing Qdrant environment variables")
    sys.exit(1)

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY
)

print(f"Connecting to Qdrant at {QDRANT_URL}:{QDRANT_PORT}")
print(f"Using collection: {QDRANT_COLLECTION}")

# Create test data
point_id = str(uuid.uuid4())
print(f"Creating test point with ID: {point_id}")

# Create wavelet hash vector (256-bit, alternating 0s and 1s)
whash_vector = [float(i % 2) for i in range(256)]

# Create sparse vector for tags
tag_indices = [0, 5, 10, 15, 20]  # Some example tag indices
tag_values = [0.9, 0.8, 0.7, 0.6, 0.5]  # Corresponding confidence scores

# Create tag dictionary for human readability
tags = {
    "person": 0.9,
    "building": 0.8,
    "sky": 0.7, 
    "tree": 0.6,
    "car": 0.5
}

# Insert point with the new format
try:
    print("Inserting point...")
    # Insert with just the primary vector
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[{
            "id": point_id,
            "vector": whash_vector,  # wavelet hash as primary vector
            "payload": {
                "file": "test_image.jpg",
                "gcs_uri": "gs://test-bucket/test_image.jpg",
                "https_uri": "https://storage.cloud.google.com/test-bucket/test_image.jpg",
                "timestamp": "2023-01-01T00:00:00Z",
                "tags": tags,  # Include human-readable tags in payload
                # Store sparse vector data in payload since we can't directly use sparse vectors
                "_sparse_vectors": {
                    "camie": {
                        "indices": tag_indices,
                        "values": tag_values
                    }
                }
            }
        }]
    )
    
    print("Point inserted successfully")
except Exception as e:
    print(f"Error inserting point: {e}")
    sys.exit(1)

# Retrieve the point to verify
try:
    print("\nRetrieving point...")
    result = client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True
    )
    
    if result:
        point = result[0]
        print(f"Retrieved point with ID: {point.id}")
        print(f"Vector dimensions: {len(point.vector)}")
        print(f"First 10 elements of vector: {point.vector[:10]}")
        
        # Check sparse vectors stored in payload
        if "_sparse_vectors" in point.payload:
            print("\nSparse vectors (from payload):")
            for name, sparse_vector in point.payload["_sparse_vectors"].items():
                print(f"  {name}: {len(sparse_vector['indices'])} values")
                print(f"  Indices: {sparse_vector['indices']}")
                print(f"  Values: {sparse_vector['values']}")
        
        # Check payload
        print("\nPayload:")
        for key, value in point.payload.items():
            if key == "tags":
                print(f"  {key}: {len(value)} tags")
                # Print first 5 tags
                for i, (tag, score) in enumerate(sorted(value.items(), key=lambda x: x[1], reverse=True)):
                    if i < 5:
                        print(f"    {tag}: {score}")
            else:
                print(f"  {key}: {value}")
        
        print("\nTest successful - point was inserted and retrieved correctly")
    else:
        print("Error: Could not retrieve point")
except Exception as e:
    print(f"Error retrieving point: {e}")
    
# Test search functionality
try:
    print("\nTesting search...")
    # Create a modified version of the original hash to test similarity search
    # Flip a few bits to simulate a similar but not identical hash
    search_vector = whash_vector.copy()
    for i in range(10):  # Flip 10 bits
        search_vector[i*25] = 1.0 - search_vector[i*25]
    
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=search_vector,
        limit=5,
        with_payload=True
    )
    
    print(f"Found {len(results)} search results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result.id}, Score: {result.score:.4f}")
        print(f"     File: {result.payload.get('file', 'unknown')}")
        
except Exception as e:
    print(f"Error searching: {e}")

print("\nTest completed")