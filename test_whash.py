#!/usr/bin/env python
"""
Test script for wavelet hash implementation.
"""
import os
import sys
import base64
import argparse
from pathlib import Path
from io import BytesIO

from PIL import Image
import imagehash
from vector_generator import (
    calculate_wavelet_hash, 
    get_image_hash_similarity,
    search_by_image,
    initialize_qdrant_client
)

def load_image(image_path):
    """Load an image file into binary data."""
    with open(image_path, 'rb') as f:
        return f.read()

def test_whash_calculation(image_path):
    """Test calculating a wavelet hash from an image."""
    # Load image data
    image_data = load_image(image_path)
    
    # Calculate hash using our function
    hash_vector = calculate_wavelet_hash(image_data)
    
    # Calculate hash directly using the imagehash library for comparison
    image = Image.open(BytesIO(image_data))
    hash_obj = imagehash.whash(image, hash_size=16)
    
    print(f"Image: {image_path}")
    print(f"Wavelet hash (imagehash): {hash_obj}")
    print(f"Hash vector length: {len(hash_vector)}")
    print(f"First 16 bits of vector: {hash_vector[:16]}")
    
    return hash_vector

def test_hash_similarity(image_path1, image_path2):
    """Test similarity calculation between two images."""
    # Load image data
    image_data1 = load_image(image_path1)
    image_data2 = load_image(image_path2)
    
    # Calculate similarity
    similarity = get_image_hash_similarity(image_data1, image_data2)
    
    print(f"Image 1: {image_path1}")
    print(f"Image 2: {image_path2}")
    print(f"Similarity score: {similarity:.4f} (higher is more similar)")
    
    return similarity

def test_qdrant_search(image_path, limit=5):
    """Test searching for similar images in Qdrant."""
    # Load image data
    image_data = load_image(image_path)
    
    # Search for similar images
    results = search_by_image(image_data, limit=limit)
    
    print(f"Search results for {image_path}:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result['id']}")
        print(f"     Score: {result['score']:.4f}")
        print(f"     File: {result['payload'].get('file', 'unknown')}")
        
        # Print top 5 tags
        tags = result['payload'].get('tags', {})
        if tags:
            print("     Top tags:")
            sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]
            for tag, score in sorted_tags:
                print(f"       {tag}: {score:.4f}")
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test wavelet hash implementation")
    parser.add_argument("--image", type=str, help="Path to test image", required=False)
    parser.add_argument("--compare", type=str, help="Path to second image for comparison", required=False)
    parser.add_argument("--search", action="store_true", help="Search for similar images in Qdrant")
    parser.add_argument("--limit", type=int, default=5, help="Limit for search results")
    
    args = parser.parse_args()
    
    if args.image:
        # Make sure the image exists
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            sys.exit(1)
            
        # Test hash calculation
        hash_vector = test_whash_calculation(args.image)
        
        # If a second image is provided, test similarity
        if args.compare:
            if not os.path.exists(args.compare):
                print(f"Error: Comparison image not found at {args.compare}")
                sys.exit(1)
                
            similarity = test_hash_similarity(args.image, args.compare)
            
        # If search is requested, test Qdrant search
        if args.search:
            results = test_qdrant_search(args.image, limit=args.limit)
    else:
        # If no image is provided, print usage
        parser.print_help()
        
if __name__ == "__main__":
    main()